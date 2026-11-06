"""台語 TTS 推論 —— 與訓練一致的唯一入口

所有推論設定都必須與訓練時一致，否則品質會被誤判成模型能力不足。
以下每一條都是實測踩過、修正後才明顯改善的：

  1. 底模：英文版 T3（pretrained_models/t3_cfg）+ 擴充詞彙 + LoRA
     —— 工具包訓練用的是英文版；套到多語版 T3 上 LoRA 修正量會失準
  2. 無語言標記（英文版本身就沒有；訓練時也沒有）
  3. 參考音 prompt 長度 75 token（= 訓練的 prompt_duration 3 秒）
     —— 用預設長度會造成「開頭重複第一個字」
  4. 句首大寫：保留預設 punc_norm（訓練時也有；拿掉反而不穩）
  5. 變調模型：輸入本調台羅，內部自動轉成變調

依賴 chatterbox-finetuning 工具包（src/），必須在該目錄下執行
（cb_server.py 會自動切換目錄並設定 sys.path）。
"""
import inspect
import torch
import perth
import time


class _NoWatermark:
    """浮水印模組在部分環境載入失敗；推論品質不受影響，直接停用。"""
    def apply_watermark(self, wav, sample_rate=None, **kw):
        return wav

    def get_watermark(self, *a, **k):
        return None


perth.PerthImplicitWatermarker = _NoWatermark

from peft import PeftModel                                   # noqa: E402
from src.config import TrainConfig                           # noqa: E402
from src.model import resize_and_load_t3_weights             # noqa: E402
from src.chatterbox_.tts import ChatterboxTTS                # noqa: E402
from src.chatterbox_.models.t3.t3 import T3                  # noqa: E402
from sandhi import convert                                   # noqa: E402

PROMPT_LEN = 75
_cfg = TrainConfig()


class TaigiTTS:
    def __init__(self, adapter, ref, sandhi=True, device="cuda"):
        tmp = ChatterboxTTS.from_local(_cfg.model_dir, device="cpu")
        state, hp = tmp.t3.state_dict(), tmp.t3.hp
        hp.text_tokens_dict_size = _cfg.new_vocab_size
        if hasattr(hp, "use_cache"):
            hp.use_cache = False
        t3 = resize_and_load_t3_weights(T3(hp=hp), state)
        del tmp, state
        t3.tfmr.config._attn_implementation = "eager"   # sdpa 不支援 output_attentions
        

        t3.hp.speech_cond_prompt_len = PROMPT_LEN
        eng = ChatterboxTTS.from_local(_cfg.model_dir, device="cpu")
        eng.t3 = PeftModel.from_pretrained(t3, adapter, is_trainable=False)
        import os

        if os.environ.get("CB_MERGE_LORA", "0") == "1":
            print("[LORA MERGE TEST] Starting merge", flush=True)

            eng.t3.merge_adapter()

            merged = sum(
                bool(getattr(module, "merged", False))
                for module in eng.t3.modules()
            )

            print(
                f"[LORA MERGE TEST] Merged modules: {merged}",
                flush=True
            )

            if merged == 0:
                raise RuntimeError(
                    "未偵測到成功合併的 LoRA 模組"
                )
        eng.t3.to(device).eval()
        eng.s3gen.to(device).eval()
        eng.ve.to(device).eval()
        eng.device = device
        eng.prepare_conditionals(
        ref,
        exaggeration=0.5
        )
        # generate() 不再傳入 audio_prompt_path
        # 它會直接重用 eng.conds
        p = inspect.signature(eng.generate).parameters

        self.kw = {
        k: v
        for k, v in {
        "exaggeration": 0.5,
        "cfg_weight": 0.5
        }.items()
        if k in p
        }
        self.eng = eng
        self.sandhi = sandhi
        self.sr = eng.sr

    def __call__(self, tailo, seed=2):
        """tailo：本調數字調台羅"""
        print("[TTS RAW]", repr(tailo), flush=True)

        text = convert(tailo) if self.sandhi else tailo

        print("[TTS FINAL]", repr(text), flush=True)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        t3 = self.eng.t3.get_base_model()

        def check_hooks(stage):
            counts = {
                i: len(t3.tfmr.layers[i].self_attn._forward_hooks)
                for i in (9, 12, 13)
            }
            print(
                f"[HOOK TEST] {stage}: {counts}",
                flush=True
            )

        check_hooks("before")

        import os
        from torch.profiler import profile, ProfilerActivity

        self._cb_profile_calls = getattr(
            self, "_cb_profile_calls", 0
        ) + 1

        if (
            os.environ.get("CB_T3_PROFILE") == "1"
            and self._cb_profile_calls == 2
        ):
            activities = [ProfilerActivity.CPU]

            if ProfilerActivity.CUDA in torch.profiler.supported_activities():
                activities.append(ProfilerActivity.CUDA)

            print("[T3 PROFILE] 開始分析第二段 TTS", flush=True)

            with profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False
            ) as prof:
                w = self.eng.generate(text, **self.kw)

            print("\n========== CPU TOP 20 ==========", flush=True)
            print(
                prof.key_averages().table(
                    sort_by="self_cpu_time_total",
                    row_limit=20
                ),
                flush=True
            )

            if ProfilerActivity.CUDA in activities:
                print("\n========== CUDA TOP 20 ==========", flush=True)
                print(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=20
                    ),
                    flush=True
                )

            trace_path = (
                "/home/tku/chatterbox-finetuning/"
                "t3_profile_trace.json"
            )

            try:
                prof.export_chrome_trace(trace_path)
                print("[T3 PROFILE] Trace:", trace_path, flush=True)
            except Exception as exc:
                print("[T3 PROFILE] Trace 匯出失敗:", exc, flush=True)

        else:
            w = self.eng.generate(text, **self.kw)

        check_hooks("after")

        return w.squeeze().detach().cpu().numpy().astype("float32")


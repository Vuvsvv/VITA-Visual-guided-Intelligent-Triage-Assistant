"""不需要 GPU / torch 就能跑的靜態回歸測試。

四個微服務要 import 就得先有 torch、transformers、BreezyVoice repo，CI 上不可能裝，
所以這裡用 AST + 原始碼檢查來釘住審查報告點名的幾個 bug，避免日後被改回去。

對應審查報告：C2、C3、C4、C5、C6、C14、C15、C16、C17、C19。
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / "services"

BREEZY_ASR = SERVICES / "breezy_asr" / "main.py"
BREEZY_TTS = SERVICES / "breezy_tts" / "main.py"
TAIWANESE_TTS = SERVICES / "taiwanese_tts" / "main.py"


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def find_func(path: Path, name: str):
    for node in ast.walk(tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{path.name} 找不到函式 {name}")


def calls_in(node) -> set[str]:
    """收集節點底下所有被呼叫的名稱（含 a.b.c 形式的最後一段與完整路徑）。"""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            names.add(ast.unparse(sub.func))
    return names


# ══════════════════════════ C3：不要卡住 event loop ══════════════════════════

@pytest.mark.parametrize("path", [BREEZY_ASR], ids=["breezy_asr"])
def test_asr_endpoint_does_not_block_event_loop(path):
    fn = find_func(path, "transcribe")
    called = calls_in(fn)
    blocking = {c for c in called if c.startswith(("librosa.", "model.generate", "asr_pipeline"))}
    if isinstance(fn, ast.AsyncFunctionDef):
        assert not blocking, (
            f"{path.parent.name}: async endpoint 直接呼叫阻塞函式 {blocking}，"
            "會卡住整個 event loop（連 health check 都會被卡）"
        )
        offloaded = {c for c in called if "run_in_executor" in c or "to_thread" in c}
        assert offloaded, f"{path.parent.name}: 沒看到 run_in_executor / to_thread"


# ══════════════════════════ C4：切句不是死碼 ══════════════════════════

def test_split_text_is_not_dead_code():
    fn = find_func(BREEZY_TTS, "split_text")
    body = [n for n in fn.body if not isinstance(n, ast.Expr)]  # 跳過 docstring
    first = body[0]
    assert not (
        isinstance(first, ast.Return) and ast.unparse(first.value) == "[text]"
    ), "split_text 第一行就 return [text]，底下 27 行切句邏輯是死碼"


def _load_split_text():
    fn = find_func(BREEZY_TTS, "split_text")
    ns: dict = {"re": re, "MAX_CHUNK_LEN": 30, "List": list}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<split_text>", "exec"), ns)
    return ns["split_text"]


def test_split_text_actually_splits():
    """把函式抽出來單獨執行，確認長文會被切開。"""
    src = BREEZY_TTS.read_text(encoding="utf-8")
    split_text = _load_split_text()
    chunks = split_text("一二三四五六七八九十。" * 8, 30)
    assert len(chunks) > 1
    assert "".join(chunks) == "一二三四五六七八九十。" * 8
    assert all(len(c) <= 30 for c in chunks)
    assert "MAX_CHUNK_LEN" in src


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "。。。"])
def test_empty_chunks_are_guarded(text):
    """
    C4 把死碼拿掉之後 split_text 就真的會回 []，
    呼叫端不能再假設「一定至少有一段」，否則 parts[0] 會 IndexError。
    """
    split_text = _load_split_text()
    assert split_text(text) in ([], ["。。。"])  # 純標點可能自成一段
    fn = find_func(BREEZY_TTS, "synthesize")
    body = ast.unparse(fn)
    assert "if not chunks" in body, "synthesize 沒有處理 split_text 回空 list 的情況"
    stream = find_func(BREEZY_TTS, "api_synthesize_stream")
    assert "text.strip()" in ast.unparse(stream), "串流端點沒有擋空字串"


# ══════════════════════════ C5：log 不可以說謊 ══════════════════════════

def test_cudnn_log_matches_reality():
    src = BREEZY_TTS.read_text(encoding="utf-8")
    assert not re.search(r"cudnn\.benchmark\s*=\s*False", src), (
        "把 cudnn.benchmark 設成 False 之後卻 log「已啟用」"
    )


# ══════════════════════════ C6：WAV 量化要 clip ══════════════════════════

def test_wav_quantisation_clips():
    path = BREEZY_TTS
    fn = find_func(path, "waveform_to_wav_bytes")
    assert any("clip" in c for c in calls_in(fn)), (
        "waveform 超過 ±1.0 時 int16 會溢位翻號，聽起來就是爆音；乘 32767 前要先 np.clip"
    )


def test_clipping_behaviour_is_correct():
    import numpy as np

    waveform = np.array([1.4, -1.6, 0.5], dtype=np.float32)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    assert pcm.tolist() == [32767, -32767, 16383]


# ══════════════════════════ C19：speed 不可以被靜默吃掉 ══════════════════════════

@pytest.mark.parametrize(
    "path",
    [BREEZY_ASR, BREEZY_TTS, TAIWANESE_TTS],
    ids=["breezy_asr", "breezy_tts", "taiwanese_tts"],
)
def test_internal_failures_return_5xx(path):
    """
    C18：網關只看 status_code，下游若把內部錯誤包成 200 + success:false，
    整條鏈上的 status code 語意在最常見的失敗路徑上還是壞的。
    """
    src = path.read_text(encoding="utf-8")
    assert "status_code=500" in src, f"{path.parent.name} 的例外處理仍然回 200"


def test_speed_detection_does_not_accept_kwargs():
    """C19：**kwargs 不能算「支援 speed」—— 參數照收但可能被丟掉，就是靜默忽略。"""
    fn = find_func(BREEZY_TTS, "_detect_speed_support")
    assert "VAR_KEYWORD" not in ast.unparse(fn)


def test_speed_reaches_synthesis():
    path = BREEZY_TTS
    src = path.read_text(encoding="utf-8")
    fn = find_func(path, "synthesize")
    assert "speed" in [a.arg for a in fn.args.args], (
        "TTSRequest 有 speed 欄位、網關白名單也放行，但 synthesize() 根本沒收這個參數"
    )
    assert "request.speed" in src or "speed=speed" in src


# ══════════════════════════ C14：m4a 路徑與錯誤處理 ══════════════════════════

def test_m4a_path_and_ffmpeg_errors():
    src = TAIWANESE_TTS.read_text(encoding="utf-8")
    fn = find_func(TAIWANESE_TTS, "to_m4a_bytes")
    unparsed = ast.unparse(fn)
    assert '.replace(\'.wav\', \'.m4a\')' not in unparsed, (
        "字串 replace 會誤傷路徑中其他 .wav 片段，請用 Path.with_suffix"
    )
    assert "with_suffix" in unparsed
    assert "capture_output=True" in unparsed, "ffmpeg 失敗時 stderr 全被丟掉了"
    assert "returncode" in unparsed or "stderr" in unparsed
    assert "check=True" not in unparsed
    assert src  # sanity


# ══════════════════════════ C15：Gemini 呼叫不可以阻塞 ══════════════════════════

def test_gemini_call_is_offloaded():
    fn = find_func(TAIWANESE_TTS, "chinese_to_tailo")
    assert isinstance(fn, ast.AsyncFunctionDef), "chinese_to_tailo 應該是 async 並把同步呼叫丟到 thread"
    called = calls_in(fn)
    assert any("to_thread" in c for c in called) or any(
        "generate_content_async" in c for c in called
    ), "model.generate_content() 是同步的，直接在 async endpoint 裡呼叫會卡住整個服務"


# ══════════════════════════ C16 / C17：型別註記與無效 global ══════════════════════════

def test_cb_synthesize_return_annotation_is_a_tuple():
    fn = find_func(TAIWANESE_TTS, "cb_synthesize")
    ann = ast.unparse(fn.returns) if fn.returns else ""
    assert ann.startswith("tuple") or ann.startswith("Tuple"), (
        f"實際回傳 (wav, sr)，但註記是 {ann!r}"
    )


def test_synthesize_all_return_annotation_is_specific():
    fn = find_func(TAIWANESE_TTS, "synthesize_all")
    ann = ast.unparse(fn.returns) if fn.returns else ""
    assert ann not in ("tuple", "Tuple", ""), "-> tuple 太模糊"
    assert "[" in ann


def test_no_useless_global_declaration():
    fn = find_func(TAIWANESE_TTS, "synthesize_all")
    for node in ast.walk(fn):
        if isinstance(node, ast.Global):
            assert "SAMPLE_RATE" not in node.names, "宣告了 global SAMPLE_RATE 卻從未賦值"


# ══════════════════════════ v5.0.0：台羅前處理（Chatterbox）══════════════════════════
# Chatterbox 只認台羅，漢字一個都不能漏進去；這幾個純函式從 main.py 抽出來單獨執行，
# 一樣不需要 import 整個服務（不需要 Gemini 套件、numpy、soundfile）。

def _load_pure(names, extra=None):
    import typing
    import unicodedata
    mod = tree(TAIWANESE_TTS)
    keep = []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            keep.append(node)
    ns = {"re": re, "unicodedata": unicodedata, "List": typing.List}
    ns.update(extra or {})
    exec(compile(ast.Module(body=keep, type_ignores=[]), "main_subset", "exec"), ns)
    return ns


_NORM = ["_HANZI_RE", "_PUNCT_MAP", "_TONE_MARKS", "_MARKED_SYL_RE",
         "has_hanzi", "_marks_to_digits", "normalize_tailo"]


def test_normalize_tailo_cleans_gemini_output():
    ns = _load_pure(_NORM)
    norm, has_hanzi = ns["normalize_tailo"], ns["has_hanzi"]
    # 全形標點、引號、調號（Gemini 偶爾不照指示）都要整理成訓練時的格式
    assert norm("「thâu-khak thiànn，kian3-gi7 li2」") == "thau5-khak4 thiann3, kian3-gi7 li2"
    assert norm("  li2   ho2 ？ ") == "li2 ho2?"
    assert norm("kian3-gi7 li2 khi3 khuann3 lai7-kho1.") == "kian3-gi7 li2 khi3 khuann3 lai7-kho1."
    assert has_hanzi("li2 去 khuann3") and not has_hanzi("li2 khi3 khuann3")


def test_split_for_tts_respects_limit_and_never_cuts_a_word():
    ns = _load_pure(["split_for_tts"], {"MAX_SEG_LEN": 70})
    split = ns["split_for_tts"]
    text = ("bok8-tsian5 gua2 kan1-na1 e7-tang3 tsiau3 li2 kong2 e5 tsing5-hing5, "
            "tau3 li2 tshue7 sik4-hap8 e5 kua3-ho7 kho1-piat8, be7-tang3 kan1-na1 tui3-ue7 "
            "to7 phuan3-tuan3 kam2 giam5-tiong7. m7-koh4 li2 tsit4-ma2 iau2 kiann5 e7 tang7, "
            "ma7 bo5 kong2-tioh8 u7 bing5-hian2 gua7-siong1. ping5-siong5 siann2-mih8 "
            "si5-tuann7 khah4 hong1-pian7?")
    segs = split(text)
    assert len(segs) > 1 and all(len(s) <= 70 for s in segs), segs
    assert " ".join(segs) == text, "拆段後接回去必須跟原文一模一樣（沒有掉字、沒有切斷詞）"
    # 兩句加起來沒超過上限就不要拆（太短的段落自迴歸模型反而不穩）
    assert split("liau2-kai2. tsit4-ma2 iau2 e7-tang3 kiann5-loo7 bo5?") == [
        "liau2-kai2. tsit4-ma2 iau2 e7-tang3 kiann5-loo7 bo5?"]


def test_replacements_match_whole_words_only():
    ns = _load_pure(["apply_replacements"], {"load_replace_map": lambda: {"ho7": "XX"}})
    # ho7 是 kua3-ho7（掛號）的一部分，不能被誤換
    assert ns["apply_replacements"]("kua3-ho7 ho7.") == "kua3-ho7 XX."


def test_engine_rejects_hanzi_and_main_validates_before_sending():
    """漢字送進 Chatterbox 一定念不出來：引擎要擋，main.py 在送出前也要先擋。"""
    src = TAIWANESE_TTS.read_text(encoding="utf-8")
    engine = (SERVICES / "taiwanese_tts" / "engine" / "cb_server.py").read_text(encoding="utf-8")
    assert "has_hanzi(tailo)" in src, "skip_gemini 模式沒有檢查呼叫端送來的文字"
    assert "has_hanzi(out)" in src, "Gemini 的輸出沒有檢查是否夾帶漢字"
    assert "_CJK.search(text)" in engine, "引擎沒有拒收漢字"

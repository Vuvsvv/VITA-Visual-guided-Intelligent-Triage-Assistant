"""設定檔 / 部署腳本 / 文件的一致性測試。

對應審查報告：C20、C21、P1、P2、P3、P4、P5、P6、P7、P8。
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

CONFIG_LOCAL = ROOT / "services_config.yaml"
CONFIG_DOCKER = ROOT / "services_config_docker.yaml"
CONFIG_VPS = ROOT / "services_config_vps.yaml"
COMPOSE = ROOT / "docker-compose.yml"
DEPLOY = ROOT / "deploy.sh"
START_ALL = ROOT / "start_all.sh"
README = ROOT / "README.md"
ENV_EXAMPLE = ROOT / ".env.example"
TEST_CLIENT = ROOT / "test_client.py"

ALL_CONFIGS = [CONFIG_LOCAL, CONFIG_DOCKER, CONFIG_VPS]


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose() -> dict:
    return load(COMPOSE)


# ══════════════════════════ C20：timeout 要對得起來 ══════════════════════════

def test_nginx_timeout_covers_the_slowest_service():
    deploy = DEPLOY.read_text(encoding="utf-8")
    read_timeout = int(re.search(r"proxy_read_timeout\s+(\d+)s", deploy).group(1))
    send_timeout = int(re.search(r"proxy_send_timeout\s+(\d+)s", deploy).group(1))
    slowest = max(
        s.get("timeout", 60)
        for path in ALL_CONFIGS
        for s in load(path)["services"].values()
    )
    assert read_timeout >= slowest, (
        f"Nginx proxy_read_timeout {read_timeout}s < 服務 timeout {slowest}s → 長文合成一定 504"
    )
    assert send_timeout >= slowest


def test_engine_timeout_matches_gateway_timeout():
    """台語 TTS 內部呼叫 Chatterbox 引擎的 timeout 不該比網關給它的還長。"""
    tts = load(CONFIG_LOCAL)["services"]["taiwanese_tts"]
    src = (ROOT / "services" / "taiwanese_tts" / "main.py").read_text(encoding="utf-8")
    cb_timeout = float(re.search(r'CB_TIMEOUT",\s*"(\d+)"', src).group(1))
    assert cb_timeout <= tts["timeout"]


# ══════════════════════════ C21：ASR 容器要有 GPU ══════════════════════════

def test_asr_container_reserves_a_gpu(compose):
    for name in ("taiwanese-asr",):
        devices = (
            compose["services"][name]
            .get("deploy", {})
            .get("resources", {})
            .get("reservations", {})
            .get("devices", [])
        )
        assert any(d.get("driver") == "nvidia" for d in devices), (
            f"{name} 沒有分配 GPU，Whisper 系模型在 CPU 上會慢到不能用"
        )


# ══════════════════════════ P1：模型名要跟實作一致 ══════════════════════════

@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_taiwanese_tts_config_reflects_chatterbox(path):
    tts = load(path)["services"]["taiwanese_tts"]
    blob = f"{tts['description']} {' '.join(tts['models'])}"
    assert "GPT-SoVITS" not in blob and "BigVGAN" not in blob, (
        f"{path.name}：v5.0.0 已改用 Chatterbox，設定檔還在報舊模型給呼叫端"
    )
    assert "Chatterbox" in blob
    assert tts["version"].startswith("5."), "版本號還停在 4.x"


def test_readme_reflects_chatterbox():
    readme = README.read_text(encoding="utf-8")
    assert "Chatterbox" in readme
    assert "BigVGAN" not in readme, "README 架構圖/表格還是 VITS+BigVGAN"
    assert "api_v2.py" not in readme, "README 還在教人啟動 GPT-SoVITS api_v2"


# ══════════════════════════ P2：compose 少一個服務 ══════════════════════════

def test_compose_starts_every_service_the_config_enables(compose):
    """docker 設定檔標成 enabled 的服務，compose 就要真的把它起起來。"""
    enabled = {k for k, v in load(CONFIG_DOCKER)["services"].items() if v["enabled"]}
    container_of = {
        "taiwanese_asr": "taiwanese-asr",
        "taiwanese_tts": "taiwanese-tts",
        "chinese_tts": "chinese-tts",
    }
    for svc in enabled:
        assert container_of[svc] in compose["services"], (
            f"{svc} 被標成 enabled，但 docker-compose.yml 沒有這個服務 → /health 永遠 unhealthy"
        )


def test_docker_config_points_at_container_names_not_the_host():
    """P2/P4：Docker 版設定不該再繞回宿主機。"""
    for name, svc in load(CONFIG_DOCKER)["services"].items():
        if svc["enabled"]:
            assert "host.docker.internal" not in svc["endpoint"], name


# ══════════════════════════ P3：systemd unit ══════════════════════════

def test_systemd_units_point_at_modules_that_exist():
    deploy = DEPLOY.read_text(encoding="utf-8")
    for module in re.findall(r"uvicorn\s+(services\.[\w.]+):app", deploy):
        rel = Path(module.replace(".", "/") + ".py")
        assert (ROOT / rel).exists(), f"deploy.sh 指向不存在的模組 {module} → ModuleNotFoundError"


def test_services_are_importable_packages():
    assert (ROOT / "services" / "__init__.py").exists()
    for pkg in ("breezy_asr", "breezy_tts", "taiwanese_tts"):
        assert (ROOT / "services" / pkg / "__init__.py").exists(), f"{pkg} 缺 __init__.py"


def test_deploy_creates_a_unit_for_every_enabled_service():
    deploy = DEPLOY.read_text(encoding="utf-8")
    units = set(re.findall(r"/etc/systemd/system/(ai-[\w-]+)\.service", deploy))
    assert "ai-chinese-tts" in units, "設定檔把 chinese_tts 標成 enabled，deploy.sh 卻沒建 unit"
    for unit in units:
        assert re.search(rf"systemctl enable[^\n]*\b{unit}\b", deploy), f"{unit} 沒有被 enable"


BREEZY_TTS_DIR = ROOT / "services" / "breezy_tts"


def _breezy_tts_module() -> str:
    """Docker 的 CMD 指向 breezy_tts 的哪一個模組。"""
    dockerfile = (BREEZY_TTS_DIR / "Dockerfile").read_text(encoding="utf-8")
    return re.search(r'uvicorn"?,?\s*"?(\w+):app', dockerfile).group(1)


def test_each_service_has_exactly_one_implementation():
    """
    S3：`breezy_tts` 曾經同時有 main.py（3.x）與 main_v2.py（4.x），
    而 Dockerfile 部署的是舊的那份。每個服務只留一個 main.py，這種事才不會再發生。
    """
    for pkg in ("breezy_asr", "breezy_tts", "taiwanese_tts"):
        mains = sorted(p.name for p in (ROOT / "services" / pkg).glob("main*.py"))
        assert mains == ["main.py"], f"services/{pkg}/ 有多份實作: {mains}"
    assert not list((ROOT / "services").rglob("*.bak")), "還有 .bak 備份檔，請用 git tag/branch"
    assert not (ROOT / "services" / "taiwanese_tts_old").exists()


def test_breezy_tts_entrypoints_all_agree():
    """三條啟動路徑（Docker / start_all.sh / systemd）必須指向同一個模組。"""
    entrypoints = {
        "Dockerfile": _breezy_tts_module(),
        "start_all.sh": re.search(
            r"python services/breezy_tts/(\w+)\.py", START_ALL.read_text(encoding="utf-8")
        ).group(1),
        "deploy.sh": re.search(
            r"uvicorn services\.breezy_tts\.(\w+):app", DEPLOY.read_text(encoding="utf-8")
        ).group(1),
    }
    assert len(set(entrypoints.values())) == 1, f"三條啟動路徑跑不同檔案: {entrypoints}"

    module = next(iter(entrypoints.values()))
    assert (BREEZY_TTS_DIR / f"{module}.py").exists()
    dockerfile = (BREEZY_TTS_DIR / "Dockerfile").read_text(encoding="utf-8")
    copied = " ".join(re.findall(r"^COPY (.+)$", dockerfile, re.MULTILINE))
    assert f"{module}.py" in copied, f"Dockerfile 沒有 COPY {module}.py，映像檔裡不會有這一份"


def test_chinese_tts_config_version_matches_the_running_module():
    """設定檔回報的版本要跟實際跑的那份自報的版本一致（P1 就是這個毛病）。"""
    module = _breezy_tts_module()
    src = (BREEZY_TTS_DIR / f"{module}.py").read_text(encoding="utf-8")
    actual = re.search(r'version\s*=\s*"([\d.]+)"', src).group(1)
    for path in ALL_CONFIGS:
        declared = load(path)["services"]["chinese_tts"]["version"]
        assert declared == actual, (
            f"{path.name} 報 chinese_tts {declared}，但實際跑的 {module}.py 是 {actual}"
        )


# ══════════════════════════ 網路暴露面 ══════════════════════════

def test_only_the_gateway_port_is_published(compose):
    """
    8001–8004 的微服務完全沒有 API Key 檢查 —— 對外 publish 等於讓人繞過網關
    直接呼叫模型（:8003 背後是 Gemini，會直接花掉額度）。
    網關走 Docker 內部網路的容器名稱找它們，不經過宿主機的 port，綁 loopback 不影響運作。
    """
    for name, svc in compose["services"].items():
        for mapping in svc.get("ports", []):
            published = str(mapping)
            if name == "gateway":
                continue
            assert published.startswith("127.0.0.1:"), (
                f"{name} 把 {published} 開到所有網卡上，但它沒有任何認證"
            )


def _readme_section(heading: str) -> str:
    body = README.read_text(encoding="utf-8").split(heading, 1)[1]
    return body.split("\n### ", 1)[0].split("\n## ", 1)[0]


def test_local_path_keeps_engine_on_loopback():
    """
    本機路徑（start_all.sh）的台語 TTS 就跑在宿主機上，
    Chatterbox 引擎綁 127.0.0.1 就夠 —— 沒有理由多開一個沒有認證的 port。
    """
    section = _readme_section("### 方式 A")
    assert "CB_HOST=127.0.0.1" in section and "0.0.0.0" not in section
    assert "CB_HOST=127.0.0.1" in START_ALL.read_text(encoding="utf-8")


def test_docker_path_binds_engine_for_containers_and_firewalls_it():
    """
    容器裡的台語 TTS 透過 host.docker.internal 連宿主機，對容器來說宿主機算「外部」，
    所以引擎必須綁 0.0.0.0 —— 而綁了就一定要用防火牆擋住 9881。
    """
    section = _readme_section("### 方式 B")
    assert "CB_HOST=0.0.0.0" in section, "Docker 路徑沒教人改 bind，容器會連不到引擎"
    assert "ufw deny 9881" in section, "教人開放 9881 卻沒教人擋起來"


def test_every_open_bind_instruction_carries_a_firewall_warning():
    """任何叫人把引擎綁 0.0.0.0 的地方，附近都要提醒擋掉 9881。"""
    tts_dir = ROOT / "services" / "taiwanese_tts"
    for path in (README, START_ALL, tts_dir / "main.py", tts_dir / "README.md",
                 tts_dir / "engine" / "cb_server.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            # 只看引擎的 bind 設定；服務自己在容器裡 uvicorn host="0.0.0.0" 是正常的
            if "CB_HOST=0.0.0.0" not in line:
                continue
            window = "\n".join(lines[max(0, i - 8):i + 9])
            assert "9881" in window and ("防火牆" in window or "ufw" in window), (
                f"{path.name}:{i + 1} 叫人開放綁定卻沒提防火牆"
            )


def test_deploy_firewall_blocks_internal_ports():
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert re.search(r"ufw deny \$\{port\}|ufw deny", deploy), "deploy.sh 沒有明確擋下內部 port"
    denied = re.search(r"for port in ([\d\s]+); do", deploy)
    assert denied, "找不到 deny 的 port 清單"
    ports = set(denied.group(1).split())
    assert {"8000", "8001", "8002", "8003", "8004", "9881"} <= ports, ports


def test_deploy_does_not_reset_the_users_firewall():
    """S13/部署安全：ufw --force reset 會清掉 VPS 上既有的所有規則。"""
    executable = [
        line
        for line in DEPLOY.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not [l for l in executable if "ufw --force reset" in l]


def test_deploy_does_not_echo_the_api_key():
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert not re.search(r'echo[^\n]*\$\{?API_KEY\}?', deploy), (
        "API Key 被 echo 到終端機，會留在 screen buffer / CI log"
    )


# ══════════════════════════ P4：VPS 設定檔 ══════════════════════════

def test_deploy_installs_a_bare_metal_config():
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert CONFIG_VPS.name in deploy, "deploy.sh 複製的設定檔裡有 host.docker.internal，VPS 上解析不了"
    assert CONFIG_VPS.exists()


def test_vps_config_has_no_docker_hostnames():
    for name, svc in load(CONFIG_VPS)["services"].items():
        assert "host.docker.internal" not in svc["endpoint"], name
        assert "127.0.0.1" in svc["endpoint"] or "localhost" in svc["endpoint"], name


def test_endpoints_can_be_overridden_by_env():
    """P4 的長期解：endpoint 可以完全走環境變數，不用維護第三份設定。"""
    src = (ROOT / "gateway.py").read_text(encoding="utf-8")
    assert "_ENDPOINT" in src


# ══════════════════════════ P7：台語 TTS 可以容器化 ══════════════════════════

def test_taiwanese_tts_has_a_dockerfile_and_compose_service(compose):
    assert (ROOT / "services" / "taiwanese_tts" / "Dockerfile").exists(), (
        "台語 TTS 只是一個 HTTP client（不碰 torch），可以直接 Docker 化"
    )
    svc = compose["services"]["taiwanese-tts"]
    env = svc.get("environment", [])
    joined = " ".join(env) if isinstance(env, list) else " ".join(f"{k}={v}" for k, v in env.items())
    assert "CB_API" in joined and "host.docker.internal:9881" in joined, (
        "容器裡的台語 TTS 要透過 host.docker.internal 連宿主機的 Chatterbox 引擎 :9881"
    )


def test_taiwanese_tts_requirements_have_no_torch():
    reqs = (ROOT / "services" / "taiwanese_tts" / "requirements.txt").read_text(encoding="utf-8")
    assert "torch" not in reqs


# ══════════════════════════ P5 / P6：範例與欄位 ══════════════════════════

def test_test_client_uses_fields_that_actually_exist():
    """P5：範例用了 ASR 回應裡不存在的欄位，照抄的後端會 KeyError。"""
    client_src = TEST_CLIENT.read_text(encoding="utf-8")
    assert "traditional_chinese" not in client_src, (
        "TranscriptionResponse 沒有 traditional_chinese 這個欄位"
    )
    asr_src = (ROOT / "services" / "breezy_asr" / "main.py").read_text(encoding="utf-8")
    assert "traditional_chinese" not in asr_src


def test_deprecated_field_is_documented():
    """P6：tailo_romanization 現在裝的是台語漢字，要標 deprecated 並指引新欄位。"""
    tts_src = (ROOT / "services" / "taiwanese_tts" / "main.py").read_text(encoding="utf-8")
    assert "deprecated" in tts_src.lower()
    assert "taigi_hanji" in README.read_text(encoding="utf-8")
    gateway_src = (ROOT / "gateway.py").read_text(encoding="utf-8")
    assert "台羅拼音" not in gateway_src, "網關 docstring 還在說會回台羅拼音"


# ══════════════════════════ P8：文件小錯 ══════════════════════════

def test_env_example_does_not_claim_asr_needs_gemini():
    """P8：台語 ASR（breezy_asr）完全沒用到 Gemini，只有 TTS 用。"""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    idx = next(i for i, l in enumerate(lines) if l.startswith("GEMINI_API_KEY="))
    comment = lines[idx - 1]
    assert comment.lstrip().startswith("#")
    assert "ASR/TTS" not in comment, f"註解說 ASR 也需要 Gemini：{comment}"
    assert "TTS" in comment
    asr_src = (ROOT / "services" / "breezy_asr" / "main.py").read_text(encoding="utf-8")
    assert "genai" not in asr_src and "GEMINI" not in asr_src


def test_readme_project_structure_matches_the_tree():
    readme = README.read_text(encoding="utf-8")
    structure = readme.split("## 專案結構", 1)[1]
    for entry in sorted(p.name for p in (ROOT / "services").iterdir() if p.is_dir() and not p.name.startswith("_")):
        assert entry in structure, f"專案結構沒有列出 services/{entry}/"


def test_start_all_step_numbering_matches_reality():
    src = START_ALL.read_text(encoding="utf-8")
    steps = re.findall(r"\[(\d+)/(\d+)\]", src)
    assert steps, "找不到步驟標號"
    total = {t for _, t in steps}
    assert len(total) == 1, f"步驟總數不一致: {total}"
    total = int(total.pop())
    assert len(steps) == total, f"印了 {len(steps)} 個步驟但標號寫 /{total}"
    assert sorted(int(n) for n, _ in steps) == list(range(1, total + 1))

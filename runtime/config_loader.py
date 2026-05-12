import os
import yaml
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

def load() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Орчны хувьсагч config дахь утгыг дарна
    env_token = os.environ.get("CHIMEGE_TOKEN", "")
    if env_token:
        cfg["chimege_token"] = env_token

    if not cfg.get("chimege_token"):
        raise EnvironmentError(
            "CHIMEGE_TOKEN тохируулаагүй байна.\n"
            "  PowerShell: $env:CHIMEGE_TOKEN='<token>'\n"
            "  эсвэл config.yaml-д chimege_token: '<token>' гэж бич."
        )

    return cfg

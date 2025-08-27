from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError as e:
    raise RuntimeError("Python 3.11 以降が必要です（標準の tomllib を使用）。") from e


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Dictの深いマージ（後勝ち）。"""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_toml(path: Path) -> Dict[str, Any]:
    """TOMLを読み込んでdictを返す。"""
    with path.open("rb") as f:
        return tomllib.load(f)


def _get_profile(argv: list[str]) -> Optional[str]:
    """--profile / 環境変数からプロファイル名を取得。"""
    for i, tok in enumerate(argv):
        if tok.startswith("--profile="):
            return tok.split("=", 1)[1].strip()
        if tok == "--profile" and i + 1 < len(argv):
            return argv[i + 1].strip()
    return os.getenv("APP_PROFILE") or os.getenv("PROFILE")


def _require(cfg: Dict[str, Any], dotted: str) -> None:
    """必須キー存在チェック（ドット区切り）。"""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing config key: {dotted}")
        cur = cur[key]


def load_config() -> Dict[str, Any]:
    """
    設定のロードと正規化。
    読み込み順（後勝ち）:
      1) config/config.default.toml
      2) config/profiles/<profile>.toml または config/properties/<profile>.toml（存在すれば）
      3) config/config.local.toml（存在すれば）
    さらに一部の便利デフォルトを埋めて返す。
    """
    # modules/config_loader.py から見たプロジェクトルート
    root = Path(__file__).resolve().parent.parent
    cfg_dir = root / "config"

    cfg: Dict[str, Any] = {}

    # 1) default
    default = cfg_dir / "config.default.toml"
    if default.exists():
        cfg = deep_merge(cfg, _load_toml(default))

    # 2) profile/properties（任意）
    profile = _get_profile(sys.argv)
    if profile:
        for folder in ("profiles", "properties"):
            p = cfg_dir / folder / f"{profile}.toml"
            if p.exists():
                cfg = deep_merge(cfg, _load_toml(p))
                break

    # 3) local（任意）
    local = cfg_dir / "config.local.toml"
    if local.exists():
        cfg = deep_merge(cfg, _load_toml(local))

    # 必須キー（sub_region は各ブロックでフォールバックするため必須化しない）
    for key in (
        "app.window_title",
        "screen.monitor_index",
        "screen.region",
        "screen.sub_region",
        "match.template_path",
        "match.threshold",
        "timing.interval_ms",
    ):
        _require(cfg, key)

    # 参照しやすいように一部のデフォルトを埋める（tomlに未記載でも動く）
    cfg.setdefault("app", {}).setdefault("debug_save", True)
    cfg.setdefault("app", {}).setdefault("debug_dir", "debug_shots")
    cfg.setdefault("match", {}).setdefault("scales", [1.00])
    cfg.setdefault("match", {}).setdefault("required_consecutive_hits", 1)
    cfg.setdefault("input", {}).setdefault("click_button", "left")
    cfg.setdefault("input", {}).setdefault("move_before_click", True)
    cfg.setdefault("input", {}).setdefault("move_duration_sec", 0.05)
    cfg.setdefault("input", {}).setdefault("click_debounce_ms", 700)
    cfg.setdefault("input", {}).setdefault("post_click_wait_ms", 400)

    # circle デフォルト
    circ = cfg.setdefault("circle", {})
    circ.setdefault("enabled", False)
    circ.setdefault("method", "hough")  # "hough" | "template" | "hybrid"
    circ.setdefault("threshold", 0.90)
    circ.setdefault("min_radius", 8)
    circ.setdefault("max_radius", 26)
    circ.setdefault("dp", 1.2)
    circ.setdefault("param1", 100)
    circ.setdefault("param2", 28)
    circ.setdefault("min_dist", 25)
    circ.setdefault("required_consecutive_hits", cfg["match"].get("required_consecutive_hits", 1))
    circ.setdefault("template_path", "templates/circle.png")
    circ.setdefault("mask_path", "templates/circle_mask.png")
    circ.setdefault("scales", list(cfg["match"].get("scales", [1.00])))
    # ※ circ["sub_region"] は設定側で任意（無ければ後でフォールバック）

    # match_secondary デフォルト
    ms = cfg.setdefault("match_secondary", {})
    ms.setdefault("enabled", False)
    ms.setdefault("template_path", "templates/target_secondary.png")
    ms.setdefault("cut_template_path", "templates/target_secondary_cut.png")  # 将来拡張用
    ms.setdefault("threshold", 0.88)
    ms.setdefault("scales", [1.00, 1.10, 1.20])
    ms.setdefault("required_consecutive_hits", 1)
    # ※ ms["sub_region"] は設定側で任意（無ければ後でフォールバック）

    # 便利用: コンフィグディレクトリを覚えておく（GUIの「設定フォルダを開く」で使用）
    cfg["_meta"] = {"config_dir": str(cfg_dir)}
    return cfg

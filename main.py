# ./main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
エントリポイント（GUI分離版）

目的:
- 設定とロガーの初期化のみを担い、UI本体は ui/main_window.py へ委譲する。
- これにより、GUIあり運用とヘッドレス運用の切替・テスト容易性を高める。

背景:
- 旧来の main.py には GUI 生成処理が存在しなかった（＝ウィンドウが出ない）:contentReference[oaicite:12]{index=12}。
- gui_actions.py は GUI ボタンから呼ぶべき開始／停止等の処理を提供している:contentReference[oaicite:13]{index=13}。
- app.window_title 等の GUI 向け設定が config で管理されている:contentReference[oaicite:14]{index=14}。
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict

# ------------------------------------------
# パス救済（プロジェクトルート / modules / ui を sys.path へ）
# ------------------------------------------
ROOT = Path(__file__).resolve().parent
for p in [ROOT, ROOT / "modules", ROOT / "ui"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# ------------------------------------------
# ロガー取得（あれば app_logger、無ければ標準 logging）
# ------------------------------------------
try:
    from modules.app_logger import get_app_logger  # 既存があれば利用
    log = get_app_logger()
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y:%m:%d %H:%M:%S",
    )
    log = logging.getLogger("main")

# ------------------------------------------
# 設定読み込み（config_loader があれば利用、無ければ簡易TOML読取）
# ------------------------------------------
def _load_toml_simple(path: Path) -> Dict[str, Any]:
    """Python 3.11+ は tomllib、未満は tomli を使って TOML を読み込む簡易関数。"""
    if not path.exists():
        return {}
    try:
        if sys.version_info >= (3, 11):
            import tomllib  # type: ignore
        else:
            import tomli as tomllib  # type: ignore
        with path.open("rb") as f:
            return dict(tomllib.load(f))
    except Exception as e:
        log.warning("TOML 読み込みに失敗: %s (%r)", path, e)
        return {}

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """浅・深混在の再帰マージユーティリティ。"""
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out

def load_config() -> Dict[str, Any]:
    """
    設定を読み込み、config.default.toml → config.toml の順でマージして返す。
    modules/config_loader が存在すれば、そちらの API を優先利用する。
    """
    # 1) 既存のローダがあれば利用
    try:
        from modules.config_loader import load_config as _loader  # type: ignore
        cfg = _loader()
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    # 2) フォールバック（簡易ロード）
    default = ROOT / "config" / "config.default.toml"
    local   = ROOT / "config" / "config.toml"
    cfg = {}
    cfg = _deep_merge(cfg, _load_toml_simple(default))
    cfg = _deep_merge(cfg, _load_toml_simple(local))
    return cfg

# ------------------------------------------
# UI 起動を委譲
# ------------------------------------------
def main() -> int:
    """設定とロガーを初期化し、ui/main_window.launch_ui を呼び出すだけの薄い main。"""
    log.info("アプリ起動: GUI モードへ移行します")
    config = load_config()
    try:
        from ui.main_window import launch_ui
    except Exception as e:
        log.exception("UI モジュールの読み込みに失敗しました: %r", e)
        return 2
    try:
        return int(launch_ui(config=config, log=log))  # 正常終了=0
    except Exception as e:
        log.exception("UI 起動で例外が発生しました: %r", e)
        return 1
    finally:
        log.info("アプリ終了")

# ------------------------------------------
# スクリプト実行
# ------------------------------------------
if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import json
import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

# 既存ロガーを利用（cp932・日次ファイル・Tokyo時刻・[YYYY:MM:DD hh:mm:ss][LEVEL]）
from modules.app_logger import get_app_logger  # type: ignore

# 既存の設定ローダで messages の場所を参照しても良いが、
# このモジュール単体でも動くようにデフォルトを持たせる
DEFAULT_MESSAGES_PATHS = [
    Path("config/messages.ja.json"),
    Path("config/messages.json"),
]

@dataclass
class MessageDef:
    template: str
    level: str = "INFO"

class _Catalog:
    """JSONメッセージカタログの軽量ローダ（必要に応じてホットリロード）。"""
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._mtime: float = -1.0
        self._map: Dict[str, MessageDef] = {}

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._map = {}
                self._mtime = -1.0
                return
            mtime = self.path.stat().st_mtime
            if mtime == self._mtime:
                return  # 変化なし
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            new_map: Dict[str, MessageDef] = {}
            for code, v in data.items():
                tpl = v.get("template", "")
                lvl = v.get("level", "INFO")
                if tpl:
                    new_map[code] = MessageDef(template=tpl, level=lvl)
            self._map = new_map
            self._mtime = mtime

    def get(self, code: str) -> Optional[MessageDef]:
        self._load()
        return self._map.get(code)

# ---- シングルトン的に使う（cfgからpathを差し込めるようにもしておく） ----
_CATALOG: Optional[_Catalog] = None
_CATALOG_PATH: Optional[Path] = None

def configure_messages(path: Optional[str | Path]) -> None:
    """明示的にメッセージファイルを指定（任意）。"""
    global _CATALOG, _CATALOG_PATH
    _CATALOG = None
    _CATALOG_PATH = Path(path) if path else None

def _resolve_messages_path() -> Optional[Path]:
    # 明示指定があればそれを使う
    if _CATALOG_PATH:
        return _CATALOG_PATH
    # 既定の候補を順に探索
    for p in DEFAULT_MESSAGES_PATHS:
        if p.exists():
            return p
    return None

def _get_catalog() -> Optional[_Catalog]:
    global _CATALOG
    if _CATALOG is None:
        p = _resolve_messages_path()
        if p:
            _CATALOG = _Catalog(p)
    return _CATALOG

# ---- API ----
def format_message(code: str, **params: Any) -> Tuple[str, str]:
    """
    カタログからコードを引いて整形文言と推奨レベルを返す。
    ヒットしない場合はフォールバック（コードをそのまま出す）。
    Returns: (message_text, suggested_level)
    """
    cat = _get_catalog()
    if cat:
        md = cat.get(code)
        if md:
            try:
                return md.template.format(**params), md.level
            except Exception as e:
                # パラメータ欠落などは安全側にメッセージで可視化
                return f"{code}: format error: {e}", md.level
    # フォールバック（最低限）
    if params:
        return f"{code}: {params}", "INFO"
    return code, "INFO"

def log_event(code: str, *, level: Optional[str] = None, logger=None, **params: Any) -> None:
    """
    イベントコード＋パラメータでログ出力。
    level を省略するとカタログ既定レベルを採用。
    """
    lg = logger or get_app_logger()
    text, sug = format_message(code, **params)
    use_level = (level or sug or "INFO").upper()
    # 既存のレベル名に合わせて dispatch
    if use_level == "DEBUG":
        lg.debug(text)
    elif use_level == "WARNING":
        lg.warning(text)
    elif use_level == "ERROR":
        lg.error(text)
    elif use_level == "CRITICAL":
        lg.critical(text)
    else:
        lg.info(text)

# ---- 便利ラッパ（既存 log_app_start/log_app_exit 相当をイベントコードで） ----
def log_app_start_event(title: str) -> None:
    log_event(
        "APP_START",
        title=title,
        python=platform.python_version(),
        platform=platform.platform(),
    )

def log_app_exit_event() -> None:
    log_event("APP_EXIT")

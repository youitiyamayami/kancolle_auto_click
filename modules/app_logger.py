# modules/app_logger.py
"""
アプリ全体で共通利用するロガーの初期化モジュール。

■ 目的
- どこでも `from modules.app_logger import get_logger` で同一設定のロガーを取得
- ログは「log\\YYYYMMDD_app.log」に日付ごとに追記、コンソールにも出力可能
- フォーマット: [YYYY:MM:DD hh:mm:ss][LEVEL] メッセージ
- タイムゾーンは Asia/Tokyo（ローカル時刻）
- Shift_JIS(CP932) でファイル出力（ユーザー環境要件）
- ハンドラ重複防止、未捕捉例外もERRORで記録可能
- 実行中にログレベルの動的変更も可能

■ 主なAPI
- get_logger(name="app", cfg: dict | None = None, **overrides) -> logging.Logger
- set_level(level: str | int, logger: logging.Logger | None = None) -> None
- log_job(title: str, logger: logging.Logger | None = None, level: int = logging.INFO)
    -> contextmanager（開始/終了と経過時間を自動ログ）
- install_global_excepthook(logger: logging.Logger | None = None) -> None

■ 互換API（既存 main.py 対応用）
- get_app_logger(cfg: dict | None = None) -> logging.Logger
- log_app_start(title: str = "Application") -> None
- log_app_exit() -> None
- log_info(msg: str) -> None
- log_error(msg: str) -> None
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, Optional

try:
    # 3.9+ 標準
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ==========================================================
# 設定デフォルト
# ==========================================================
@dataclass(frozen=True)
class LoggerDefaults:
    level: str = "INFO"          # 既定レベル
    console: bool = True         # コンソール出力の有無
    file: bool = True            # ファイル出力の有無
    dir: str = "log"             # ログディレクトリ
    encoding: str = "cp932"      # ファイル出力エンコーディング（Shift_JIS互換）
    timezone: str = "Asia/Tokyo" # 表示タイムゾーン
    logfile_prefix: str = "app"  # ログファイル名のプレフィックス
    logfile_ext: str = ".log"    # 拡張子（.log 推奨）


_DEFAULTS = LoggerDefaults()


# ==========================================================
# ユーティリティ
# ==========================================================
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _today_str(tz_name: str) -> str:
    tz = ZoneInfo(tz_name) if ZoneInfo and tz_name else None
    now = datetime.now(tz=tz)
    return now.strftime("%Y%m%d")


def _build_logfile_path(root: Path, prefix: str, ext: str, tz_name: str) -> Path:
    return root / f"{_today_str(tz_name)}_{prefix}{ext}"


def _to_level(level: str | int | None, fallback: int = logging.INFO) -> int:
    if level is None:
        return fallback
    if isinstance(level, int):
        return level
    name = str(level).upper()
    return getattr(logging, name, fallback)


class _TzFormatter(logging.Formatter):
    """
    タイムゾーン付きの asctime を出力するためのフォーマッタ。
    """
    def __init__(self, fmt: str, datefmt: Optional[str], tz_name: str) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._tz = ZoneInfo(tz_name) if ZoneInfo and tz_name else None

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=self._tz)
        if datefmt:
            return dt.strftime(datefmt)
        # デフォルト表記
        return dt.strftime("%Y:%m:%d %H:%M:%S")


# ==========================================================
# ハンドラ重複防止のためのタグ
# ==========================================================
_HANDLER_TAG = "kancolle_app_logger_handler_tag"


def _is_ours(handler: logging.Handler) -> bool:
    return getattr(handler, _HANDLER_TAG, False) is True


def _tag(handler: logging.Handler) -> None:
    setattr(handler, _HANDLER_TAG, True)


# ==========================================================
# 主関数: get_logger
# ==========================================================
def get_logger(
    name: str = "app",
    *,
    cfg: Optional[Dict[str, Any]] = None,
    level: Optional[str | int] = None,
    console: Optional[bool] = None,
    file: Optional[bool] = None,
    logfile_root: Optional[Path | str] = None,
    encoding: Optional[str] = None,
    timezone: Optional[str] = None,
    logfile_prefix: Optional[str] = None,
    logfile_ext: Optional[str] = None,
) -> logging.Logger:
    """
    共通ロガーを返す。複数回呼んでもハンドラは重複追加されない。
    """
    # 1) cfg と引数から最終値を決定
    d = (cfg or {}).get("logging", {}) if (cfg and "logging" in cfg) else (cfg or {})
    level_val = _to_level(level if level is not None else d.get("level", _DEFAULTS.level))
    use_console = bool(console if console is not None else d.get("console", _DEFAULTS.console))
    use_file = bool(file if file is not None else d.get("file", _DEFAULTS.file))

    root_dir = Path(
        logfile_root if logfile_root is not None else d.get("dir", _DEFAULTS.dir)
    )
    enc = encoding if encoding is not None else d.get("encoding", _DEFAULTS.encoding)
    tz_name = timezone if timezone is not None else d.get("timezone", _DEFAULTS.timezone)
    prefix = logfile_prefix if logfile_prefix is not None else d.get("logfile_prefix", _DEFAULTS.logfile_prefix)
    ext = logfile_ext if logfile_ext is not None else d.get("logfile_ext", _DEFAULTS.logfile_ext)

    # 2) フォーマット定義（要件準拠）
    fmt = "[%(asctime)s][%(levelname)s] %(message)s"
    datefmt = "%Y:%m:%d %H:%M:%S"
    formatter = _TzFormatter(fmt=fmt, datefmt=datefmt, tz_name=tz_name)

    logger = logging.getLogger(name)

    # 3) ロガー本体のレベルは DEBUG にして、ハンドラ側で絞るのが柔軟
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 祖先に伝播させない（重複防止）

    # 4) 既存ハンドラの再利用判定
    has_console = False
    has_file = False
    for h in logger.handlers:
        if _is_ours(h):
            if isinstance(h, logging.StreamHandler):
                has_console = True
            if isinstance(h, logging.FileHandler):
                has_file = True

    # 5) コンソールハンドラ
    if use_console and not has_console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level_val if level is not None else _to_level(d.get("level_console"), level_val))
        ch.setFormatter(formatter)
        _tag(ch)
        logger.addHandler(ch)

    # 6) ファイルハンドラ
    if use_file and not has_file:
        logfile_dir = _ensure_dir(Path(root_dir))
        logfile_path = _build_logfile_path(logfile_dir, prefix, ext, tz_name)
        fh = logging.FileHandler(logfile_path, mode="a", encoding=enc, delay=True)
        fh.setLevel(_to_level(d.get("level_file"), logging.DEBUG))
        fh.setFormatter(formatter)
        _tag(fh)
        logger.addHandler(fh)

    # 7) 明示レベル→主にコンソール閾値に反映
    set_level(level_val, logger=logger)

    return logger


# ==========================================================
# 便利API
# ==========================================================
def set_level(level: str | int, *, logger: Optional[logging.Logger] = None) -> None:
    """実行中にログレベルを変更する。"""
    lg = logger or logging.getLogger("app")
    new_level = _to_level(level)

    console_targets = [h for h in lg.handlers if _is_ours(h) and isinstance(h, logging.StreamHandler)]
    targets = console_targets or [h for h in lg.handlers if _is_ours(h)]

    for h in targets:
        h.setLevel(new_level)


@contextmanager
def log_job(
    title: str,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Iterator[None]:
    """開始/終了と経過秒を自動ログするコンテキストマネージャ。"""
    lg = logger or logging.getLogger("app")
    start = perf_counter()
    lg.log(level, f"{title} - 開始")
    try:
        yield
        elapsed = perf_counter() - start
        lg.log(level, f"{title} - 終了 (elapsed={elapsed:.3f}s)")
    except Exception:
        elapsed = perf_counter() - start
        lg.error(f"{title} - 失敗 (elapsed={elapsed:.3f}s)", exc_info=True)
        raise


def install_global_excepthook(logger: Optional[logging.Logger] = None) -> None:
    """未捕捉例外を ERROR でロギングする excepthook をインストールする。"""
    lg = logger or logging.getLogger("app")

    def _hook(exc_type, exc, tb):
        try:
            lg.exception("Uncaught exception", exc_info=(exc_type, exc, tb))
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


# ==========================================================
# 互換API（main.py が使っている関数名を提供）
# ==========================================================
_APP_LOGGER: Optional[logging.Logger] = None


def get_app_logger(cfg: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """アプリ共通ロガー（シングルトン）を取得。"""
    global _APP_LOGGER
    if _APP_LOGGER is None:
        _APP_LOGGER = get_logger("app", cfg=cfg)
        # 未捕捉例外も拾う
        install_global_excepthook(_APP_LOGGER)
    return _APP_LOGGER


def log_app_start(title: str = "Application") -> None:
    """起動ログ（互換）"""
    lg = get_app_logger()
    try:
        import platform
        lg.info(f"アプリ起動: {title} | Python {platform.python_version()} | {platform.platform()}")
    except Exception:
        lg.info(f"アプリ起動: {title}")


def log_app_exit() -> None:
    """終了ログ（互換）"""
    lg = get_app_logger()
    lg.info("アプリ終了")


def log_info(msg: str) -> None:
    """任意情報ログ（互換）"""
    get_app_logger().info(msg)


def log_error(msg: str) -> None:
    """任意エラーログ（互換）"""
    get_app_logger().error(msg)


# ==========================================================
# サンプル（手動テスト用）
# ==========================================================
if __name__ == "__main__":
    log_app_start("Demo")
    log_info("INFO ログのテスト")
    try:
        with log_job("デモ処理"):
            raise ValueError("テスト例外")
    except Exception:
        log_error("例外を捕捉しました")
    log_app_exit()

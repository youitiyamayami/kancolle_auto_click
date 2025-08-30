# modules/app_context.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import logging


@dataclass
class AppContext:
    """
    GUIイベントからビジネスロジックを呼び出すための依存注入用コンテキスト。
    - logger / config / project_root は読み取り専用の共通資源
    - set_status はUI反映用コールバック（Tk依存をここで止める）
    - start_worker / stop_worker / worker_is_alive はワーカー制御
    - open_path は OS 依存のファイル/フォルダオープン
    - get_config_dir は設定ディレクトリ取得
    """
    logger: logging.Logger
    config: Dict[str, Any]
    project_root: Path

    set_status: Callable[[str], Any]

    start_worker: Callable[[Callable[[str], None]], None]
    stop_worker: Callable[[], None]
    worker_is_alive: Callable[[], bool]

    open_path: Callable[[str], None]
    get_config_dir: Callable[[], Optional[str]]

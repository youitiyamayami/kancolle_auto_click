# main.py — TOML設定版（default → profile → local の後勝ちマージ）
# 依存: pip install mss opencv-python numpy pyautogui pywin32
# 動作例:
#   python main.py                   # defaultのみ
#   python main.py --profile prod    # properties/prod.toml を差分マージ
#   set APP_PROFILE=prod && python main.py  # 環境変数でプロフィール指定

from __future__ import annotations
import sys, os, threading, time
from pathlib import Path
from typing import Any, Dict, Optional

import mss
import numpy as np
import cv2
import tkinter as tk

try:
    import tomllib  # Python 3.11+
except Exception:
    raise RuntimeError("Python 3.11 以降が必要です（tomllib が標準搭載）。")

try:
    import win32gui  # ウィンドウ矩形取得（任意）
except ImportError:
    win32gui = None


# ==========================
# 設定ローダ（TOML）
# ==========================
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)

def _get_profile(argv: list[str]) -> Optional[str]:
    # --profile=prod or --profile prod / env APP_PROFILE, PROFILE
    for i, tok in enumerate(argv):
        if tok.startswith("--profile="):
            return tok.split("=", 1)[1].strip()
        if tok == "--profile" and i + 1 < len(argv):
            return argv[i + 1].strip()
    return os.getenv("APP_PROFILE") or os.getenv("PROFILE")

def _require(cfg: Dict[str, Any], dotted: str) -> None:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing config key: {dotted}")
        cur = cur[key]

def load_config() -> Dict[str, Any]:
    """
    読み込み順:
      1) config/config.default.toml
      2) config/profiles/<profile>.toml または config/properties/<profile>.toml（存在すれば）
      3) config/config.local.toml（存在すれば）
    後勝ちマージ。
    """
    root = Path(__file__).resolve().parent
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

    # 必須キーの最低限チェック
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
    cfg.setdefault("input", {}).setdefault("click_debounce_ms", 500)
    cfg.setdefault("input", {}).setdefault("post_click_wait_ms", 300)

    # 便利用: コンフィグディレクトリを覚えておく（GUIの「設定フォルダを開く」で使用）
    cfg["_meta"] = {"config_dir": str(cfg_dir)}
    return cfg


# ==========================
# DPI（Windows向け）
# ==========================
def set_dpi_awareness_windows():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # SYSTEM_DPI_AWARE
    except Exception:
        pass


# ==========================
# 位置計算ユーティリティ
# ==========================
def find_window_rect(window_title: str):
    """(left, top, right, bottom) を返す。見つからなければ None。"""
    if not window_title or not win32gui:
        return None
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        return None
    return win32gui.GetWindowRect(hwnd)  # (l,t,r,b)

def build_sub_bbox(cfg: Dict[str, Any]):
    """
    mss.grab 用 bbox（絶対座標）と、基準座標を返す。
    戻り値: (sub_bbox, base_left_top, sub_left_top)
      - sub_bbox: {"left": L, "top": T, "width": W, "height": H}  … SUB_REGION の絶対
      - base_left_top: REGION を適用した左上絶対 (base_x, base_y)
      - sub_left_top : SUB_REGION 左上の絶対 (sub_x, sub_y)
    """
    base_x, base_y, base_w, base_h = cfg["screen"]["region"]
    sub_x,  sub_y,  sub_w,  sub_h  = cfg["screen"]["sub_region"]
    win = str(cfg["app"]["window_title"]).strip()
    mon_idx = int(cfg["screen"]["monitor_index"])

    with mss.mss() as sct:
        if win:
            rect = find_window_rect(win)
            if rect is None:
                raise RuntimeError(f'ウィンドウが見つかりません: "{win}"')
            wl, wt, wr, wb = rect
            base_left = wl + base_x
            base_top  = wt + base_y
            sub_left  = base_left + sub_x
            sub_top   = base_top  + sub_y
            return (
                {"left": sub_left, "top": sub_top, "width": sub_w, "height": sub_h},
                (base_left, base_top),
                (sub_left,  sub_top),
            )
        else:
            # 画面基準
            mons = sct.monitors
            if mon_idx < 1 or mon_idx >= len(mons):
                raise ValueError(f"monitor_index が不正（1〜{len(mons)-1}）：{mon_idx}")
            m = mons[mon_idx]
            base_left = m["left"] + base_x
            base_top  = m["top"]  + base_y
            sub_left  = base_left + sub_x
            sub_top   = base_top  + sub_y
            return (
                {"left": sub_left, "top": sub_top, "width": sub_w, "height": sub_h},
                (base_left, base_top),
                (sub_left,  sub_top),
            )


# ==========================
# テンプレ照合
# ==========================
def match_best_scale(gray_roi, tmpl_gray, scales):
    """
    複数スケールでテンプレ照合し、最良スコアを返す。
    戻り値: dict(score, loc, tw, th, scale)
    """
    best = {"score": -1.0, "loc": (0, 0), "tw": 0, "th": 0, "scale": 1.0}
    for s in scales:
        tw = max(1, int(tmpl_gray.shape[1] * s))
        th = max(1, int(tmpl_gray.shape[0] * s))
        if tw < 2 or th < 2:
            continue
        resized = cv2.resize(
            tmpl_gray,
            (tw, th),
            interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC,
        )
        res = cv2.matchTemplate(gray_roi, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best["score"]:
            best = {"score": float(max_val), "loc": max_loc, "tw": tw, "th": th, "scale": s}
    return best

def save_debug_image(frame_bgra, top_left, size_wh, score, out_dir: Path):
    """ヒット位置に枠を書いてBGRで保存（PNG）"""
    x, y = top_left
    w, h = size_wh
    vis = frame_bgra.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255, 255), 2)
    cv2.putText(
        vis, f"score={score:.3f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255, 255), 2, cv2.LINE_AA
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    out = out_dir / f"hit_{int(time.time()*1000)}.png"
    cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR))


# ==========================
# ワーカースレッド
# ==========================
class CaptureWorker(threading.Thread):
    def __init__(self, cfg, on_log=None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.on_log = on_log or (lambda msg: None)
        self._stop = threading.Event()

        # 停止時保存用（最後に観測したフレームとベスト結果を保持）
        self._last_frame = None           # BGRA
        self._last_best = None            # dict(score, loc, tw, th, scale)

        # クリック運用用の内部状態
        self._hit_streak = 0
        self._last_click_ts = 0.0  # monotonic 秒

    def stop(self):
        self._stop.set()

    def run(self):
        # bbox 構築（サブ領域のみ）
        try:
            sub_bbox, base_left_top, sub_left_top = build_sub_bbox(self.cfg)
        except Exception as e:
            self.on_log(f"[ERR] bbox構築失敗: {e}")
            return

        # テンプレ読み込み（グレースケール）
        tmpl_path = Path(self.cfg["match"]["template_path"])
        template = cv2.imread(str(tmpl_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            self.on_log(f"[ERR] テンプレ読み込み失敗: {tmpl_path}")
            return
        th, tw = template.shape[:2]

        thr      = float(self.cfg["match"]["threshold"])
        dry_run  = bool(self.cfg["app"].get("dry_run", False))
        interval = max(1, int(self.cfg["timing"]["interval_ms"])) / 1000.0
        scales   = list(self.cfg["match"].get("scales", [1.00]))
        dbg_save = bool(self.cfg["app"].get("debug_save", True))
        dbg_dir  = Path(self.cfg["app"].get("debug_dir", "debug_shots"))

        req_hits    = int(self.cfg["match"].get("required_consecutive_hits", 1))
        debounce_ms = int(self.cfg["input"].get("click_debounce_ms", 500))
        post_wait_ms= int(self.cfg["input"].get("post_click_wait_ms", 300))
        click_button= str(self.cfg["input"].get("click_button", "left"))
        move_before = bool(self.cfg["input"].get("move_before_click", True))
        move_dur_sec= float(self.cfg["input"].get("move_duration_sec", 0.05))

        # 初期の「最後の結果」はテンプレサイズを使う
        self._last_best = {"score": -1.0, "loc": (0, 0), "tw": tw, "th": th, "scale": 1.0}

        self.on_log(
            f"[INFO] SUB_BBOX={sub_bbox}, tmpl={tmpl_path.name}({tw}x{th}), "
            f"thr={thr}, DRY_RUN={dry_run}, scales={scales}"
        )

        with mss.mss() as sct:
            while not self._stop.is_set():
                try:
                    # サブ領域だけをキャプチャ（BGRA）
                    frame = np.array(sct.grab(sub_bbox))
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

                    # テンプレ照合（ROI内のみ）
                    best = match_best_scale(gray, template, scales=scales)
                    score, loc, w, h, scale = best["score"], best["loc"], best["tw"], best["th"], best["scale"]

                    # 停止時保存用に保持
                    self._last_frame = frame
                    self._last_best = best

                    now = time.monotonic()

                    if score >= thr:
                        self._hit_streak += 1
                        cx = sub_left_top[0] + loc[0] + w / 2
                        cy = sub_left_top[1] + loc[1] + h / 2

                        self.on_log(f"[HIT] score={score:.3f} scale={scale:.2f} at ({cx:.1f},{cy:.1f}) streak={self._hit_streak}/{req_hits}")

                        if dbg_save:
                            save_debug_image(frame, loc, (w, h), score, dbg_dir)

                        # クリック判定（連続ヒット＆デバウンス）
                        due = (now - self._last_click_ts) * 1000.0 >= debounce_ms
                        if (self._hit_streak >= req_hits) and due and (not dry_run):
                            try:
                                import pyautogui
                                # セーフティ
                                pyautogui.FAILSAFE = True
                                pyautogui.PAUSE = 0.05

                                if move_before:
                                    pyautogui.moveTo(cx, cy, duration=move_dur_sec)
                                pyautogui.click(cx, cy, button=click_button)

                                self._last_click_ts = now
                                self.on_log(f"[CLICK] button={click_button} at ({cx:.1f},{cy:.1f})")
                                # クリック後の待機（UI反映待ち）
                                time.sleep(post_wait_ms / 1000.0)
                            except Exception as e:
                                self.on_log(f"[ERR] クリック失敗: {e}")

                            # 次のクリック判定のため連続カウンタをリセット
                            self._hit_streak = 0
                    else:
                        # ヒット継続が途切れたらリセット
                        if self._hit_streak != 0:
                            self.on_log(f"[INFO] ヒット連続が途切れました（reset）。")
                        self._hit_streak = 0
                        self.on_log(f"[CAP] {gray.shape[1]}x{gray.shape[0]} score={score:.3f} scale={scale:.2f}")

                    # ループ間隔
                    time.sleep(interval)

                except Exception as e:
                    self.on_log(f"[ERR] 例外: {e}")
                    break

        # ---- 停止時に最後のフレームを保存（スコアに関わらず）----
        try:
            if dbg_save and (self._last_frame is not None) and (self._last_best is not None):
                save_debug_image(
                    self._last_frame,
                    self._last_best["loc"],
                    (self._last_best["tw"], self._last_best["th"]),
                    self._last_best["score"],
                    dbg_dir,
                )
                self.on_log("[INFO] 停止時の最終キャプチャを debug_shots に保存しました。")
        except Exception as e:
            self.on_log(f"[WARN] 停止時デバッグ保存に失敗: {e}")


# ==========================
# GUI本体
# ==========================
class ControlWindow(tk.Tk):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.title("Game Bot Control")
        self.geometry("420x260")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self._worker = None
        self.status_var = tk.StringVar(value="停止中")

        tk.Label(self, text="ゲームボット操作", font=("Segoe UI", 14, "bold")).pack(pady=(12, 6))
        tk.Label(self, textvariable=self.status_var).pack(pady=(0, 8))

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=6, fill="x", padx=12)
        self.start_btn = tk.Button(btn_frame, text="開始", width=12, command=self.on_start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.stop_btn = tk.Button(btn_frame, text="停止", width=12, state="disabled", command=self.on_stop)
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        bottom = tk.Frame(self)
        bottom.pack(pady=6, fill="x", padx=12)
        tk.Button(bottom, text="設定フォルダを開く", command=self.open_config_folder).pack(side="left")
        tk.Button(bottom, text="終了", command=self.on_exit).pack(side="right")

        tk.Label(self, text="※ このウィンドウは常に最前面に固定されています（Escで終了）。", fg="#666").pack(pady=(4, 8))

        self.bind("<Escape>", lambda e: self.on_exit())
        self.after(1000, self.keep_topmost)

    # ---- イベント ----
    def on_start(self):
        if self._worker and self._worker.is_alive():
            self.status_var.set("既に稼働中です")
            return

        self.status_var.set("稼働中…")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        def log(msg: str):
            # ワーカー → GUIスレッドへ安全に反映
            self.after(0, self.status_var.set, msg)

        self._worker = CaptureWorker(self.cfg, on_log=log)
        self._worker.start()

    def on_stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=2.0)
            self._worker = None
        self.status_var.set("停止済み（最終キャプチャを保存）")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def open_config_folder(self):
        cfg_dir = self.cfg.get("_meta", {}).get("config_dir")
        if cfg_dir and Path(cfg_dir).exists():
            os.startfile(cfg_dir)
        else:
            os.startfile(os.path.dirname(os.path.abspath(__file__)))

    def on_exit(self):
        # 終了前にワーカーを止める
        try:
            if self._worker:
                self._worker.stop()
                self._worker.join(timeout=2.0)
        finally:
            self.destroy()

    def keep_topmost(self):
        self.attributes("-topmost", True)
        self.after(1000, self.keep_topmost)


# ==========================
# エントリーポイント
# ==========================
def main():
    if sys.platform.startswith("win"):
        set_dpi_awareness_windows()
    cfg = load_config()
    app = ControlWindow(cfg)
    app.mainloop()

if __name__ == "__main__":
    main()

# main.py
# -----------------------------------------------------------------------------
# このバージョンの追加点（重要な設計方針も含む）
# 1) 実行ログ機能をモジュール化（modules/app_logger.py）
#    - 起動時: log_app_start(app_name)
#    - 終了ボタン押下時: log_app_exit()
#    - 形式: [YYYY:MM:DD hh:mm:ss][LEVEL]実行された動作
#    - 出力: ./log/YYYYMMDD_app.log（同日内は追記・日付変化で新規）
#
# 2) 円検出（[circle]）と矩形テンプレ（[match]）に「別々の sub_region」を使えるようにした
#    - 矩形(プライマリ)は [match].sub_region を優先、なければ [screen].sub_region をフォールバック
#    - 円検出は [circle].sub_region を優先、なければ [match].sub_region → [screen].sub_region の順
# 3) 矩形テンプレの「セカンダリ判定」を追加（[match_secondary]）
#    - [match_secondary].enabled = true のときだけ動作
#    - こちらも [match_secondary].sub_region を優先、なければ [screen].sub_region
# 4) 検出順序（1ループで複数候補があっても「1クリックのみ」）
#    - 既定で「円 → プライマリ矩形 → セカンダリ矩形」の順に判定
#    - どれかがクリック条件を満たしてクリックしたら、そのループでは以降をスキップ
# 5) デバッグ保存
#    - 円は save_debug_circle（中心＋半径を描画）
#    - 矩形は save_debug_rect（矩形枠を描画）
# 6) 連続ヒットカウンタとデバウンス
#    - 円 / プライマリ / セカンダリで個別の「連続ヒット閾値（required_consecutive_hits）」を持てる
#    - クリックのデバウンスは共通（同一マウスを叩くので共通カウンタを使用）
# 7) 画像パスの解決を「スクリプトの場所」基準に変更
#    - 相対パスでも、どの作業ディレクトリから実行しても確実に読める
#    - 読み込み前に絶対パスと exists をログ出力（GUIステータス）
#
# 前提:
#   - Python 3.11+（tomllib 使用）
#   - pip install mss opencv-python pyautogui
# -----------------------------------------------------------------------------

from __future__ import annotations
import sys, os, threading, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

# ログ（モジュール化）
from modules.app_logger import log_app_start, log_app_exit

# スクリプトのあるディレクトリ（相対パス解決に使用）
ROOT = Path(__file__).resolve().parent


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
    ms.setdefault("cut_template_path", "templates/target_secondary_cut.png")  # ※本コードでは未使用（将来拡張用）
    ms.setdefault("threshold", 0.88)
    ms.setdefault("scales", [1.00, 1.10, 1.20])
    ms.setdefault("required_consecutive_hits", 1)
    # ※ ms["sub_region"] は設定側で任意（無ければ後でフォールバック）

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

def build_sub_bbox_from_subregion(cfg: dict, sub_region):
    """
    指定 sub_region（regionからの相対座標）で mss.grab 用 bbox を構築。
    戻り値: (sub_bbox, base_left_top, sub_left_top)
      - sub_bbox: {"left": L, "top": T, "width": W, "height": H}  … SUB_REGION の絶対
      - base_left_top: REGION を適用した左上絶対 (base_x, base_y)
      - sub_left_top : SUB_REGION 左上の絶対 (sub_x,  sub_y)
    """
    app = cfg["app"]; scr = cfg["screen"]
    base_x, base_y, base_w, base_h = map(int, scr["region"])
    sub_x, sub_y, sub_w, sub_h = map(int, sub_region)
    win = str(app.get("window_title") or "")

    with mss.mss() as sct:
        if win and win32gui:
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
            mons = sct.monitors
            mon_idx = int(scr["monitor_index"])
            if mon_idx < 1 or mon_idx >= len(mons):
                raise ValueError(f"monitor_index が不正（1〜{len(mons)-1}）：{mon_idx}")
            m = mons[mon_idx]
            base_left = m["left"] + base_x; base_top = m["top"] + base_y

    sub_left = base_left + sub_x; sub_top = base_top + sub_y
    return (
        {"left": sub_left, "top": sub_top, "width": sub_w, "height": sub_h},
        (base_left, base_top),
        (sub_left,  sub_top),
    )


# ==========================
# テンプレ照合（矩形）
# ==========================
def match_best_scale(gray_roi: np.ndarray, tmpl_gray: np.ndarray, scales) -> Dict[str, Any]:
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

# ==========================
# テンプレ照合（マスク付き：円テンプレ用）
# ==========================
def match_best_scale_with_mask(
    gray_roi: np.ndarray,
    tmpl_gray: np.ndarray,
    mask_gray: np.ndarray,
    scales,
) -> Dict[str, Any]:
    """
    マスク対応のテンプレ照合で最良スコアを返す。
    method は TM_CCORR_NORMED を使用（OpenCV 4.2+ でマスク対応）。
    戻り値: dict(score, loc, tw, th, scale)
    """
    best = {"score": -1.0, "loc": (0, 0), "tw": 0, "th": 0, "scale": 1.0}
    mask_bin = (mask_gray > 0).astype(np.uint8) * 255  # 0/255 に二値化
    for s in scales:
        tw = max(1, int(tmpl_gray.shape[1] * s))
        th = max(1, int(tmpl_gray.shape[0] * s))
        if tw < 2 or th < 2:
            continue
        tmpl_r = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        mask_r = cv2.resize(mask_bin,  (tw, th), interpolation=cv2.INTER_NEAREST)
        try:
            res = cv2.matchTemplate(gray_roi, tmpl_r, cv2.TM_CCORR_NORMED, mask=mask_r)
        except Exception:
            res = cv2.matchTemplate(gray_roi, tmpl_r, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best["score"]:
            best = {"score": float(max_val), "loc": max_loc, "tw": tw, "th": th, "scale": s}
    return best


# ==========================
# デバッグ描画
# ==========================
def save_debug_rect(frame_bgra: np.ndarray, top_left: Tuple[int,int], size_wh: Tuple[int,int], score: float, out_dir: Path):
    """ヒット位置に矩形枠を書いて保存（PNG, BGR）"""
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

def save_debug_circle(frame_bgra: np.ndarray, center: Tuple[int,int], radius: int, score: float, out_dir: Path):
    """ヒット位置に円を書いて保存（PNG, BGR）"""
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    vis = frame_bgra.copy()
    cv2.circle(vis, (cx, cy), r, (0, 255, 255, 255), 2)
    cv2.putText(
        vis, f"score={score:.3f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255, 255), 2, cv2.LINE_AA
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    out = out_dir / f"hit_{int(time.time()*1000)}.png"
    cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR))


# ==========================
# 円検出（Hough）
# ==========================
def detect_circle_hough(gray_roi: np.ndarray, circle_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    HoughCircles で円を検出し、最初の円を返す（必要に応じ最大半径優先等に変更可）。
    戻り値: {"cx": float, "cy": float, "r": float, "score": float}
    """
    blur = cv2.GaussianBlur(gray_roi, (5,5), 1.2)
    dp        = float(circle_cfg.get("dp", 1.2))
    minDist   = int(circle_cfg.get("min_dist", 25))
    param1    = int(circle_cfg.get("param1", 100))
    param2    = int(circle_cfg.get("param2", 28))
    minRadius = int(circle_cfg.get("min_radius", 8))
    maxRadius = int(circle_cfg.get("max_radius", 26))

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0,0]
    # スコアは簡易に 1.0（必要なら周縁エッジ強度などで算出可）
    return {"cx": float(x), "cy": float(y), "r": float(r), "score": 1.0}


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
        # last_best は dict: {"mode": "rect"|"circle", ...}
        self._last_best: Optional[Dict[str, Any]] = None

        # クリック運用用の内部状態（各検出で独立の連続ヒットカウンタ）
        self._hit_streak_circle = 0
        self._hit_streak_match1 = 0
        self._hit_streak_match2 = 0
        self._last_click_ts = 0.0  # monotonic 秒（共通デバウンス）

    def stop(self):
        self._stop.set()

    def run(self):
        # ------------------------------------------------------------
        # 1) ROI 構築：円 / プライマリ / セカンダリ で別々の sub_region を採用
        # ------------------------------------------------------------
        # フォールバック優先順位に注意（コメント参照）

        # (A) プライマリ矩形
        sr_match = self.cfg["match"].get("sub_region", self.cfg["screen"].get("sub_region"))
        if sr_match is None:
            raise KeyError("プライマリ矩形の sub_region が見つかりません（[match].sub_region か [screen].sub_region を設定）")
        sub_bbox_m, base_lt, sub_lt_m = build_sub_bbox_from_subregion(self.cfg, sr_match)

        # (B) 円検出（circle → match → screen の順でフォールバック）
        sr_circle = (
            self.cfg.get("circle", {}).get(
                "sub_region",
                self.cfg["match"].get("sub_region", self.cfg["screen"].get("sub_region"))
            )
        )
        if sr_circle is None:
            raise KeyError("円検出の sub_region が見つかりません（[circle]/[match]/[screen] のいずれかに設定）")
        sub_bbox_c, _, sub_lt_c = build_sub_bbox_from_subregion(self.cfg, sr_circle)

        # (C) セカンダリ矩形（オプション）
        use_secondary = bool(self.cfg.get("match_secondary", {}).get("enabled", False))
        if use_secondary:
            sr_match2 = self.cfg["match_secondary"].get("sub_region", self.cfg["screen"].get("sub_region"))
            if sr_match2 is None:
                raise KeyError("セカンダリ矩形の sub_region が見つかりません（[match_secondary].sub_region か [screen].sub_region を設定）")
            sub_bbox_m2, _, sub_lt_m2 = build_sub_bbox_from_subregion(self.cfg, sr_match2)
        else:
            sub_bbox_m2 = sub_lt_m2 = None

        self.on_log(f"[INFO] ROI match={sr_match} circle={sr_circle}" + (f" secondary={sr_match2}" if use_secondary else ""))

        # ------------------------------------------------------------
        # 2) コンフィグ読み込み（閾値など）
        # ------------------------------------------------------------
        dry_run   = bool(self.cfg["app"].get("dry_run", False))
        interval  = max(1, int(self.cfg["timing"]["interval_ms"])) / 1000.0
        dbg_save  = bool(self.cfg["app"].get("debug_save", True))
        dbg_dir   = Path(self.cfg["app"].get("debug_dir", "debug_shots"))

        # 入力系（共通デバウンス）
        debounce_ms  = int(self.cfg["input"].get("click_debounce_ms", 700))
        post_wait_ms = int(self.cfg["input"].get("post_click_wait_ms", 400))
        click_button = str(self.cfg["input"].get("click_button", "left"))
        move_before  = bool(self.cfg["input"].get("move_before_click", True))
        move_dur_sec = float(self.cfg["input"].get("move_duration_sec", 0.05))

        # 円設定
        circle_cfg   = self.cfg.get("circle", {})
        circle_on    = bool(circle_cfg.get("enabled", False))
        circle_method= str(circle_cfg.get("method", "hough")).lower()
        circle_thr   = float(circle_cfg.get("threshold", 0.90))
        circle_scales= list(circle_cfg.get("scales", [1.00]))
        req_hits_circle = int(circle_cfg.get("required_consecutive_hits", 1))

        # プライマリ矩形設定
        thr_match    = float(self.cfg["match"]["threshold"])
        scales_match = list(self.cfg["match"].get("scales", [1.00]))
        req_hits_m1  = int(self.cfg["match"].get("required_consecutive_hits", 1))

        # セカンダリ矩形設定
        if use_secondary:
            thr_match2    = float(self.cfg["match_secondary"]["threshold"])
            scales_match2 = list(self.cfg["match_secondary"].get("scales", [1.00]))
            req_hits_m2   = int(self.cfg["match_secondary"].get("required_consecutive_hits", 1))

        # ------------------------------------------------------------
        # 3) テンプレ画像の読み込み（存在しない場合はその検出系をスキップ）
        #    ★ 相対パスは ROOT（main.py の場所）を起点に解決し、exists をログ表示
        # ------------------------------------------------------------
        # プライマリ矩形テンプレ
        p1 = Path(self.cfg["match"]["template_path"])
        tmpl_path1 = p1 if p1.is_absolute() else (ROOT / p1)
        self.on_log(f"[INFO] primary template path = {tmpl_path1} (exists={tmpl_path1.exists()})")
        template1 = cv2.imread(str(tmpl_path1), cv2.IMREAD_GRAYSCALE)
        if template1 is None:
            self.on_log(f"[WARN] プライマリテンプレの読み込みに失敗: {tmpl_path1}（この検出はスキップ）")

        # セカンダリ矩形テンプレ（有効なら）
        if use_secondary:
            p2 = Path(self.cfg["match_secondary"]["template_path"])
            tmpl_path2 = p2 if p2.is_absolute() else (ROOT / p2)
            self.on_log(f"[INFO] secondary template path = {tmpl_path2} (exists={tmpl_path2.exists()})")
            template2 = cv2.imread(str(tmpl_path2), cv2.IMREAD_GRAYSCALE)
            if template2 is None:
                self.on_log(f"[WARN] セカンダリテンプレの読み込みに失敗: {tmpl_path2}（この検出はスキップ）")
        else:
            template2 = None

        # 円テンプレ（template/hybrid の場合のみ）
        circle_tmpl = circle_mask = None
        if circle_on and circle_method in ("template", "hybrid"):
            cp = Path(circle_cfg.get("template_path", "templates/circle.png"))
            mp = Path(circle_cfg.get("mask_path", "templates/circle_mask.png"))
            ct_path = cp if cp.is_absolute() else (ROOT / cp)
            mk_path = mp if mp.is_absolute() else (ROOT / mp)
            self.on_log(f"[INFO] circle tmpl = {ct_path} (exists={ct_path.exists()}), mask = {mk_path} (exists={mk_path.exists()})")
            circle_tmpl = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
            circle_mask = cv2.imread(str(mk_path), cv2.IMREAD_GRAYSCALE)
            if circle_tmpl is None or circle_mask is None:
                self.on_log(f"[WARN] 円テンプレ/マスクの読み込みに失敗: {ct_path}, {mk_path}（houghにフォールバック）")
                circle_method = "hough"

        # ------------------------------------------------------------
        # 4) メインループ
        # ------------------------------------------------------------
        with mss.mss() as sct:
            while not self._stop.is_set():
                try:
                    now = time.monotonic()

                    # クリックが発生したら、このループの後続検出はスキップするためのフラグ
                    clicked_this_loop = False

                    # ------------------------------
                    # (I) 円検出（優先度1）
                    # ------------------------------
                    if circle_on and not clicked_this_loop:
                        # 円用 ROI をキャプチャ
                        frame_c = np.array(sct.grab(sub_bbox_c))
                        gray_c  = cv2.cvtColor(frame_c, cv2.COLOR_BGRA2GRAY)

                        hit = False
                        cx_abs = cy_abs = None
                        score = -1.0
                        if circle_method == "hough":
                            circ = detect_circle_hough(gray_c, circle_cfg)
                            if circ:
                                hit = True
                                cx = circ["cx"]; cy = circ["cy"]; r = circ["r"]; score = circ["score"]
                                cx_abs = sub_lt_c[0] + cx
                                cy_abs = sub_lt_c[1] + cy
                                self._last_frame = frame_c
                                self._last_best  = {"mode":"circle","score":score,"center":(int(cx),int(cy)),"r":int(r)}
                                if dbg_save:
                                    save_debug_circle(frame_c, (cx, cy), r, score, dbg_dir)
                        elif circle_method == "template":
                            best = match_best_scale_with_mask(gray_c, circle_tmpl, circle_mask, circle_scales)
                            score = best["score"]
                            if score >= circle_thr:
                                hit = True
                                x, y = best["loc"]; w, h = best["tw"], best["th"]
                                cx = x + w/2; cy = y + h/2; r = min(w,h)/2
                                cx_abs = sub_lt_c[0] + cx
                                cy_abs = sub_lt_c[1] + cy
                                self._last_frame = frame_c
                                self._last_best  = {"mode":"circle","score":score,"center":(int(cx),int(cy)),"r":int(r)}
                                if dbg_save:
                                    save_debug_circle(frame_c, (cx, cy), r, score, dbg_dir)
                        else:  # hybrid
                            circ = detect_circle_hough(gray_c, circle_cfg)
                            if circ:
                                best = match_best_scale_with_mask(gray_c, circle_tmpl, circle_mask, circle_scales)
                                score = best["score"]
                                x, y = best["loc"]; w, h = best["tw"], best["th"]
                                cx_t = x + w/2; cy_t = y + h/2
                                if score >= circle_thr:
                                    dx = (circ["cx"] - cx_t); dy = (circ["cy"] - cy_t)
                                    dist = (dx*dx + dy*dy) ** 0.5
                                    if dist <= max(4.0, circ["r"]*0.35):
                                        hit = True
                                        cx_abs = sub_lt_c[0] + cx_t
                                        cy_abs = sub_lt_c[1] + cy_t
                                        self._last_frame = frame_c
                                        self._last_best  = {"mode":"circle","score":score,"center":(int(cx_t),int(cy_t)),"r":int(circ["r"])}
                                        if dbg_save:
                                            save_debug_circle(frame_c, (cx_t, cy_t), circ["r"], score, dbg_dir)

                        if hit:
                            self._hit_streak_circle += 1
                            self.on_log(f"[CIRCLE_HIT] score={score:.3f} at ({cx_abs:.1f},{cy_abs:.1f}) streak={self._hit_streak_circle}/{req_hits_circle}")
                            due = (now - self._last_click_ts) * 1000.0 >= debounce_ms
                            if (self._hit_streak_circle >= req_hits_circle) and due and (not dry_run) and (cx_abs is not None):
                                try:
                                    import pyautogui
                                    pyautogui.FAILSAFE = True
                                    pyautogui.PAUSE = 0.05
                                    if move_before:
                                        pyautogui.moveTo(cx_abs, cy_abs, duration=move_dur_sec)
                                    pyautogui.click(cx_abs, cy_abs, button=click_button)
                                    self._last_click_ts = now
                                    self.on_log(f"[CLICK] (circle) button={click_button} at ({cx_abs:.1f},{cy_abs:.1f})")
                                    time.sleep(post_wait_ms / 1000.0)
                                    clicked_this_loop = True
                                except Exception as e:
                                    self.on_log(f"[ERR] クリック失敗: {e}")
                                self._hit_streak_circle = 0
                        else:
                            if self._hit_streak_circle != 0:
                                self.on_log("[INFO] 円ヒット連続が途切れました（reset）。")
                            self._hit_streak_circle = 0

                    # ------------------------------
                    # (II) プライマリ矩形（優先度2）
                    # ------------------------------
                    if (template1 is not None) and (not clicked_this_loop):
                        frame_m = np.array(sct.grab(sub_bbox_m))
                        gray_m  = cv2.cvtColor(frame_m, cv2.COLOR_BGRA2GRAY)
                        best = match_best_scale(gray_m, template1, scales=scales_match)
                        score, loc, w, h, scale = best["score"], best["loc"], best["tw"], best["th"], best["scale"]

                        self._last_frame = frame_m
                        self._last_best  = {"mode":"rect","score":score,"loc":loc,"tw":w,"th":h,"scale":scale}

                        if score >= thr_match:
                            self._hit_streak_match1 += 1
                            cx = sub_lt_m[0] + loc[0] + w/2
                            cy = sub_lt_m[1] + loc[1] + h/2
                            self.on_log(f"[HIT] score={score:.3f} at ({cx:.1f},{cy:.1f}) streak={self._hit_streak_match1}/{req_hits_m1}")
                            if dbg_save:
                                save_debug_rect(frame_m, loc, (w, h), score, dbg_dir)
                            due = (now - self._last_click_ts) * 1000.0 >= debounce_ms
                            if (self._hit_streak_match1 >= req_hits_m1) and due and (not dry_run):
                                try:
                                    import pyautogui
                                    pyautogui.FAILSAFE = True
                                    pyautogui.PAUSE = 0.05
                                    if move_before:
                                        pyautogui.moveTo(cx, cy, duration=move_dur_sec)
                                    pyautogui.click(cx, cy, button=click_button)
                                    self._last_click_ts = now
                                    self.on_log(f"[CLICK] button={click_button} at ({cx:.1f},{cy:.1f})")
                                    time.sleep(post_wait_ms / 1000.0)
                                    clicked_this_loop = True
                                except Exception as e:
                                    self.on_log(f"[ERR] クリック失敗: {e}")
                                self._hit_streak_match1 = 0
                        else:
                            if self._hit_streak_match1 != 0:
                                self.on_log(f"[INFO] ヒット連続が途切れました（reset）。")
                            self._hit_streak_match1 = 0

                    # ------------------------------
                    # (III) セカンダリ矩形（優先度3 / オプション）
                    # ------------------------------
                    if use_secondary and (template2 is not None) and (not clicked_this_loop):
                        frame_m2 = np.array(sct.grab(sub_bbox_m2))
                        gray_m2  = cv2.cvtColor(frame_m2, cv2.COLOR_BGRA2GRAY)
                        best2 = match_best_scale(gray_m2, template2, scales=scales_match2)
                        score2, loc2, w2, h2, scale2 = best2["score"], best2["loc"], best2["tw"], best2["th"], best2["scale"]

                        self._last_frame = frame_m2
                        self._last_best  = {"mode":"rect","score":score2,"loc":loc2,"tw":w2,"th":h2,"scale":scale2}

                        if score2 >= thr_match2:
                            self._hit_streak_match2 += 1
                            cx2 = sub_lt_m2[0] + loc2[0] + w2/2
                            cy2 = sub_lt_m2[1] + loc2[1] + h2/2
                            self.on_log(f"[HIT2] score={score2:.3f} at ({cx2:.1f},{cy2:.1f}) streak={self._hit_streak_match2}/{req_hits_m2}")
                            if dbg_save:
                                save_debug_rect(frame_m2, loc2, (w2, h2), score2, dbg_dir)
                            due = (now - self._last_click_ts) * 1000.0 >= debounce_ms
                            if (self._hit_streak_match2 >= req_hits_m2) and due and (not dry_run):
                                try:
                                    import pyautogui
                                    pyautogui.FAILSAFE = True
                                    pyautogui.PAUSE = 0.05
                                    if move_before:
                                        pyautogui.moveTo(cx2, cy2, duration=move_dur_sec)
                                    pyautogui.click(cx2, cy2, button=click_button)
                                    self._last_click_ts = now
                                    self.on_log(f"[CLICK] (secondary) button={click_button} at ({cx2:.1f},{cy2:.1f})")
                                    time.sleep(post_wait_ms / 1000.0)
                                    clicked_this_loop = True
                                except Exception as e:
                                    self.on_log(f"[ERR] クリック失敗: {e}")
                                self._hit_streak_match2 = 0
                        else:
                            if self._hit_streak_match2 != 0:
                                self.on_log(f"[INFO] セカンダリのヒット連続が途切れました（reset）。")
                            self._hit_streak_match2 = 0

                    # ループ間隔
                    time.sleep(interval)

                except Exception as e:
                    self.on_log(f"[ERR] 例外: {e}")
                    break

        # ---- 停止時の最終デバッグ保存 ----
        try:
            if dbg_save and (self._last_frame is not None) and (self._last_best is not None):
                if self._last_best.get("mode") == "circle":
                    c = self._last_best
                    save_debug_circle(
                        self._last_frame,
                        c["center"], c["r"],
                        float(c.get("score", 0.0)),
                        dbg_dir,
                    )
                else:
                    r = self._last_best
                    save_debug_rect(
                        self._last_frame,
                        r.get("loc",(0,0)),
                        (int(r.get("tw",0)), int(r.get("th",0))),
                        float(r.get("score", 0.0)),
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

    def on_start(self):
        if self._worker and self._worker.is_alive():
            self.status_var.set("既に稼働中です")
            return

        self.status_var.set("稼働中…")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        def log(msg: str):
            # ワーカー → GUI スレッドへ安全に反映
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
            # 実行時間付きで終了ログ（モジュール）
            log_app_exit()
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
    app_name = str(cfg.get("app", {}).get("window_title", "")) or "Game Bot Control"
    # 起動ログ（モジュール）
    log_app_start(app_name)
    app = ControlWindow(cfg)
    app.mainloop()

if __name__ == "__main__":
    main()
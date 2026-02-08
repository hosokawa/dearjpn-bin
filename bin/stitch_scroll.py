#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np


def preprocess_for_match(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def row_profile(edges: np.ndarray, x0: int, x1: int) -> np.ndarray:
    # 1行ごとのエッジ量を 1D 特徴量として使う
    return edges[:, x0:x1].sum(axis=1).astype(np.float32)


def match_template_edges(
    top_img_bgr: np.ndarray,
    bottom_img_bgr: np.ndarray,
    search_height: int = 900,
    template_height: int = 220,
    min_score: float = 0.35,
    crop_x_ratio: float = 0.15,
    search_offset: int = 0,
    template_offset: int = 0,
    debug: bool = False,
):
    """
    上画像(top)の行プロファイル窓と、下画像(bottom)の行プロファイル窓を
    1px単位で比較し、最良一致を探す。

    テンプレート候補の開始位置:
      topの下端から search_height 上の点を起点に、下端方向へ1px単位で走査

    戻り値:
      (score, y_abs, t_start, th)
        score  : マッチスコア
        y_abs  : bottom画像でのマッチ開始Y（絶対座標）
        t_start: top画像でテンプレを切り出した開始Y（絶対座標）
        th     : 実際に使ったテンプレ高さ
    """
    Ht, Wt = top_img_bgr.shape[:2]
    Hb, Wb = bottom_img_bgr.shape[:2]
    if Wt != Wb:
        raise ValueError(f"Width mismatch: top={Wt}, bottom={Wb}")

    th = min(template_height, Ht, Hb)
    if th < 30:
        return 0.0, -1, -1, th

    x0 = int(Wt * crop_x_ratio)
    x1 = int(Wt * (1.0 - crop_x_ratio))
    if x1 - x0 < 80:
        x0, x1 = 0, Wt

    top_edges = preprocess_for_match(top_img_bgr)
    bottom_edges = preprocess_for_match(bottom_img_bgr)
    top_prof = row_profile(top_edges, x0, x1)
    bottom_prof = row_profile(bottom_edges, x0, x1)

    # top側テンプレ候補の走査範囲:
    # 下端から search_height 上を起点に、下端まで 1px ずつ。
    t_min = Ht - int(search_height) - int(template_offset)
    t_min = int(np.clip(t_min, 0, Ht - th))
    t_max = Ht - th

    # bottom側探索範囲:
    # 上端から 1px ずつ（search_offset があればその位置を先頭にする）
    s0 = int(np.clip(search_offset, 0, Hb - th))
    bottom_scan = bottom_prof[s0:Hb]
    if bottom_scan.shape[0] < th:
        return 0.0, -1, -1, th

    # sliding window を使って 1D 窓を行列化し、正規化相関をまとめて計算
    top_windows = np.lib.stride_tricks.sliding_window_view(top_prof, th)[t_min:t_max + 1]
    bottom_windows = np.lib.stride_tricks.sliding_window_view(bottom_scan, th)

    if top_windows.size == 0 or bottom_windows.size == 0:
        return 0.0, -1, -1, th

    top_w = top_windows.astype(np.float32, copy=False)
    bottom_w = bottom_windows.astype(np.float32, copy=False)

    top_mean = top_w.mean(axis=1, keepdims=True)
    bottom_mean = bottom_w.mean(axis=1, keepdims=True)
    top_z = top_w - top_mean
    bottom_z = bottom_w - bottom_mean

    eps = np.float32(1e-6)
    top_norm = np.sqrt(np.sum(top_z * top_z, axis=1, keepdims=True)) + eps
    bottom_norm = np.sqrt(np.sum(bottom_z * bottom_z, axis=1, keepdims=True)) + eps

    top_n = top_z / top_norm
    bottom_n = bottom_z / bottom_norm

    # score_matrix[i, j] = corr(top_start=t_min+i, bottom_start=s0+j)
    score_matrix = top_n @ bottom_n.T
    flat_idx = int(np.argmax(score_matrix))
    i, j = np.unravel_index(flat_idx, score_matrix.shape)
    best_score = float(score_matrix[i, j])
    best_t_start = int(t_min + i)
    best_y_abs = int(s0 + j)

    if debug:
        print(
            f"    score={best_score:.3f} y_abs={best_y_abs}  "
            f"t_start={best_t_start}  th={th}  "
            f"scan_top={t_min}..{t_max} scan_bottom={s0}..{Hb-th}",
            file=sys.stderr
        )

    return float(best_score), int(best_y_abs), int(best_t_start), int(th)


def stitch_two_images_by_cut(
    top: np.ndarray,
    bottom: np.ndarray,
    *,
    search_height: int,
    template_height: int,
    min_score: float,
    crop_x_ratio: float,
    search_offset: int,
    template_offset: int,
    debug: bool,
) -> np.ndarray:
    """
    あなたの方式：
      - top は「テンプレより下」を捨てる → top[:t_start]
      - bottom は「マッチより上」を捨てる → bottom[y_abs:]
      - 縦結合
    """
    score, y_abs, t_start, th = match_template_edges(
        top, bottom,
        search_height=search_height,
        template_height=template_height,
        min_score=min_score,
        crop_x_ratio=crop_x_ratio,
        search_offset=search_offset,
        template_offset=template_offset,
        debug=debug
    )

    if score < min_score or y_abs < 0 or t_start < 0:
        if debug:
            print("    (no good match) -> simple append", file=sys.stderr)
        return np.vstack([top, bottom])

    # 事故防止：極端な値は弾く（必要なら調整してOK）
    Hb = bottom.shape[0]
    if not (20 <= y_abs <= int(0.95 * Hb)):
        if debug:
            print(f"    (reject y_abs={y_abs}) -> simple append", file=sys.stderr)
        return np.vstack([top, bottom])

    top_keep = top[:t_start, :]
    bottom_keep = bottom[y_abs:, :]

    if debug:
        print(f"    cut: top_keep=0..{t_start-1}, bottom_keep={y_abs}..end", file=sys.stderr)

    return np.vstack([top_keep, bottom_keep])


def stitch_images_vertical(
    paths: list[str],
    out_path: str,
    search_height: int = 900,
    template_height: int = 220,
    min_score: float = 0.35,
    crop_x_ratio: float = 0.15,
    search_offset: int = 0,
    template_offset: int = 0,
    debug: bool = False,
) -> None:
    imgs = [cv2.imread(p) for p in paths]
    if any(im is None for im in imgs):
        bad = [p for p, im in zip(paths, imgs) if im is None]
        raise RuntimeError(f"failed to load: {bad}")

    w0 = imgs[0].shape[1]
    for p, im in zip(paths, imgs):
        if im.shape[1] != w0:
            raise ValueError(f"Width mismatch: {p} width={im.shape[1]}, expected {w0}")

    result = imgs[0]

    for idx, nxt in enumerate(imgs[1:], start=2):
        if debug:
            print(f"[{idx}]", file=sys.stderr)

        # 常に現在の連結結果(result)と次画像(nxt)を1回だけマッチして更新する。
        result = stitch_two_images_by_cut(
            result, nxt,
            search_height=search_height,
            template_height=template_height,
            min_score=min_score,
            crop_x_ratio=crop_x_ratio,
            search_offset=search_offset,
            template_offset=template_offset,
            debug=debug
        )

    if not cv2.imwrite(out_path, result):
        raise RuntimeError(f"failed to write: {out_path}")


def main():
    out_path = "output.png"
    search_h = 1000
    templ_h = 220
    min_score = 0.35
    crop_x_ratio = 0.15
    search_offset = 0
    template_offset = 0
    debug = False

    args = sys.argv[1:]
    if len(args) < 2:
        print(
            "usage: stitch_scroll.py [--out out.png] [--search H] [--template H] "
            "[--min-score S] [--crop-x R] [--search-offset PX] [--template-offset PX] [--debug] "
            "img1 img2 [img3 ...]",
            file=sys.stderr
        )
        sys.exit(1)

    paths = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--out":
            i += 1; out_path = args[i]
        elif a == "--search":
            i += 1; search_h = int(args[i])
        elif a == "--template":
            i += 1; templ_h = int(args[i])
        elif a == "--min-score":
            i += 1; min_score = float(args[i])
        elif a == "--crop-x":
            i += 1; crop_x_ratio = float(args[i])
        elif a == "--search-offset":
            i += 1; search_offset = int(args[i])
        elif a == "--template-offset":
            i += 1; template_offset = int(args[i])
        elif a == "--debug":
            debug = True
        else:
            paths.append(a)
        i += 1

    if len(paths) < 2:
        raise SystemExit("need at least 2 images")

    stitch_images_vertical(
        paths,
        out_path=out_path,
        search_height=search_h,
        template_height=templ_h,
        min_score=min_score,
        crop_x_ratio=crop_x_ratio,
        search_offset=search_offset,
        template_offset=template_offset,
        debug=debug
    )


if __name__ == "__main__":
    main()

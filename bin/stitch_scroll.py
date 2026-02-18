#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np


def preprocess_for_match(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def get_profile(edges: np.ndarray, mode: str, crop_ratio: float) -> np.ndarray:
    """
    連結方向に合わせた1次元プロファイルを抽出する。
    mode='v': 行ごとの合計 (縦方向のプロファイル)
    mode='h': 列ごとの合計 (横方向のプロファイル)
    """
    H, W = edges.shape[:2]
    if mode == 'v':
        # 左右をクロップして行プロファイルを取得
        x0 = int(W * crop_ratio)
        x1 = int(W * (1.0 - crop_ratio))
        if x1 - x0 < 80:
            x0, x1 = 0, W
        return edges[:, x0:x1].sum(axis=1).astype(np.float32)
    else:
        # 上下をクロップして列プロファイルを取得
        y0 = int(H * crop_ratio)
        y1 = int(H * (1.0 - crop_ratio))
        if y1 - y0 < 80:
            y0, y1 = 0, H
        return edges[y0:y1, :].sum(axis=0).astype(np.float32)


def match_template_edges(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    mode: str = 'v',
    search_len: int = 900,
    template_len: int = 220,
    min_score: float = 0.35,
    crop_ratio: float = 0.15,
    search_offset: int = 0,
    template_offset: int = 0,
    debug: bool = False,
):
    """
    img1 (上/左) と img2 (下/右) を比較し、最良一致を探す。
    """
    H1, W1 = img1_bgr.shape[:2]
    H2, W2 = img2_bgr.shape[:2]

    if mode == 'v':
        if W1 != W2:
            raise ValueError(f"Width mismatch: img1={W1}, img2={W2}")
        L1, L2 = H1, H2
    else:
        if H1 != H2:
            raise ValueError(f"Height mismatch: img1={H1}, img2={H2}")
        L1, L2 = W1, W2

    tl = min(template_len, L1, L2)
    if tl < 30:
        return 0.0, -1, -1, tl

    edges1 = preprocess_for_match(img1_bgr)
    edges2 = preprocess_for_match(img2_bgr)
    prof1 = get_profile(edges1, mode, crop_ratio)
    prof2 = get_profile(edges2, mode, crop_ratio)

    # img1 側のテンプレート走査範囲
    t_min = L1 - int(search_len) - int(template_offset)
    t_min = int(np.clip(t_min, 0, L1 - tl))
    t_max = L1 - tl

    # img2 側の探索範囲
    s0 = int(np.clip(search_offset, 0, L2 - tl))
    prof2_scan = prof2[s0:L2]
    if prof2_scan.shape[0] < tl:
        return 0.0, -1, -1, tl

    # Sliding window で相関計算
    win1 = np.lib.stride_tricks.sliding_window_view(prof1, tl)[t_min:t_max + 1]
    win2 = np.lib.stride_tricks.sliding_window_view(prof2_scan, tl)

    if win1.size == 0 or win2.size == 0:
        return 0.0, -1, -1, tl

    w1 = win1.astype(np.float32, copy=False)
    w2 = win2.astype(np.float32, copy=False)

    z1 = w1 - w1.mean(axis=1, keepdims=True)
    z2 = w2 - w2.mean(axis=1, keepdims=True)

    eps = np.float32(1e-6)
    n1 = z1 / (np.sqrt(np.sum(z1 * z1, axis=1, keepdims=True)) + eps)
    n2 = z2 / (np.sqrt(np.sum(z2 * z2, axis=1, keepdims=True)) + eps)

    score_matrix = n1 @ n2.T
    flat_idx = int(np.argmax(score_matrix))
    i, j = np.unravel_index(flat_idx, score_matrix.shape)
    
    best_score = float(score_matrix[i, j])
    best_t_start = int(t_min + i)
    best_match_pos = int(s0 + j)

    if debug:
        print(
            f"    mode={mode} score={best_score:.3f} match_pos={best_match_pos}  "
            f"t_start={best_t_start}  tl={tl}",
            file=sys.stderr
        )

    return best_score, best_match_pos, best_t_start, tl


def stitch_two_images_by_cut(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    mode: str,
    search_len: int,
    template_len: int,
    min_score: float,
    crop_ratio: float,
    search_offset: int,
    template_offset: int,
    debug: bool,
) -> np.ndarray:
    score, match_pos, t_start, tl = match_template_edges(
        img1, img2,
        mode=mode,
        search_len=search_len,
        template_len=template_len,
        min_score=min_score,
        crop_ratio=crop_ratio,
        search_offset=search_offset,
        template_offset=template_offset,
        debug=debug
    )

    if score < min_score or match_pos < 0 or t_start < 0:
        if debug:
            print("    (no good match) -> simple append", file=sys.stderr)
        return np.vstack([img1, img2]) if mode == 'v' else np.hstack([img1, img2])

    # 極端な値の排除
    L2 = img2.shape[0] if mode == 'v' else img2.shape[1]
    if not (20 <= match_pos <= int(0.95 * L2)):
        if debug:
            print(f"    (reject match_pos={match_pos}) -> simple append", file=sys.stderr)
        return np.vstack([img1, img2]) if mode == 'v' else np.hstack([img1, img2])

    if mode == 'v':
        keep1 = img1[:t_start, :]
        keep2 = img2[match_pos:, :]
        result = np.vstack([keep1, keep2])
    else:
        keep1 = img1[:, :t_start]
        keep2 = img2[:, match_pos:]
        result = np.hstack([keep1, keep2])

    if debug:
        print(f"    cut: keep1=0..{t_start-1}, keep2={match_pos}..end", file=sys.stderr)

    return result


def stitch_images(
    paths: list[str],
    out_path: str,
    mode: str = 'v',
    search_len: int = 900,
    template_len: int = 220,
    min_score: float = 0.35,
    crop_ratio: float = 0.15,
    search_offset: int = 0,
    template_offset: int = 0,
    debug: bool = False,
) -> None:
    imgs = [cv2.imread(p) for p in paths]
    if any(im is None for im in imgs):
        bad = [p for p, im in zip(paths, imgs) if im is None]
        raise RuntimeError(f"failed to load: {bad}")

    # 方向に応じたサイズチェック
    ref_size = imgs[0].shape[1] if mode == 'v' else imgs[0].shape[0]
    for p, im in zip(paths, imgs):
        curr_size = im.shape[1] if mode == 'v' else im.shape[0]
        if curr_size != ref_size:
            dim_name = "Width" if mode == 'v' else "Height"
            raise ValueError(f"{dim_name} mismatch: {p} size={curr_size}, expected {ref_size}")

    result = imgs[0]

    for idx, nxt in enumerate(imgs[1:], start=2):
        if debug:
            print(f"[{idx}]", file=sys.stderr)

        result = stitch_two_images_by_cut(
            result, nxt,
            mode=mode,
            search_len=search_len,
            template_len=template_len,
            min_score=min_score,
            crop_ratio=crop_ratio,
            search_offset=search_offset,
            template_offset=template_offset,
            debug=debug
        )

    if not cv2.imwrite(out_path, result):
        raise RuntimeError(f"failed to write: {out_path}")


def main():
    out_path = "output.png"
    mode = 'v'
    search_len = 1000
    templ_len = 220
    min_score = 0.35
    crop_ratio = 0.15
    search_offset = 0
    template_offset = 0
    debug = False

    args = sys.argv[1:]
    if len(args) < 2:
        print(
            "usage: stitch_scroll.py [--out out.png] [--mode v|h] [--search LEN] [--template LEN] "
            "[--min-score S] [--crop R] [--search-offset PX] [--template-offset PX] [--debug] "
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
        elif a == "--mode":
            i += 1; mode = args[i]
        elif a == "--search":
            i += 1; search_len = int(args[i])
        elif a == "--template":
            i += 1; templ_len = int(args[i])
        elif a == "--min-score":
            i += 1; min_score = float(args[i])
        elif a == "--crop" or a == "--crop-x" or a == "--crop-y":
            i += 1; crop_ratio = float(args[i])
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

    stitch_images(
        paths,
        out_path=out_path,
        mode=mode,
        search_len=search_len,
        template_len=templ_len,
        min_score=min_score,
        crop_ratio=crop_ratio,
        search_offset=search_offset,
        template_offset=template_offset,
        debug=debug
    )


if __name__ == "__main__":
    main()

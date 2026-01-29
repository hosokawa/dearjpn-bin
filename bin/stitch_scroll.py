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
    上画像(top)の下端付近(template)を、下画像(bottom)の上端付近(search)から探す。
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

    sh = min(search_height, Hb)
    th = min(template_height, Ht, sh)
    if th < 30 or sh < th + 1:
        return 0.0, -1, -1, th

    x0 = int(Wt * crop_x_ratio)
    x1 = int(Wt * (1.0 - crop_x_ratio))
    if x1 - x0 < 80:
        x0, x1 = 0, Wt

    top_edges = preprocess_for_match(top_img_bgr)
    bottom_edges = preprocess_for_match(bottom_img_bgr)

    # template: top の「下端」から template_offset だけ上にずらして取る
    t_end = max(0, Ht - int(template_offset))
    t_start = max(0, t_end - th)
    template = top_edges[t_start:t_end, x0:x1]

    # search: bottom の「上端」から search_offset だけ下を開始点にする
    s0 = int(np.clip(search_offset, 0, Hb - 1))
    s1 = min(Hb, s0 + sh)
    if s1 - s0 < th + 1:
        return 0.0, -1, -1, th

    search = bottom_edges[s0:s1, x0:x1]

    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    y_local = int(max_loc[1])
    y_abs = s0 + y_local

    if debug:
        print(f"    score={max_val:.3f} y_abs={y_abs}  t_start={t_start}  th={th}", file=sys.stderr)

    return float(max_val), int(y_abs), int(t_start), int(th)


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
    prev = imgs[0]

    for idx, nxt in enumerate(imgs[1:], start=2):
        if debug:
            print(f"[{idx}]", file=sys.stderr)

        # ここが「あなたの方式」
        merged = stitch_two_images_by_cut(
            prev, nxt,
            search_height=search_height,
            template_height=template_height,
            min_score=min_score,
            crop_x_ratio=crop_x_ratio,
            search_offset=search_offset,
            template_offset=template_offset,
            debug=debug
        )

        # result は「これまでの結果」に対して次を足す必要があるので、
        # result の末尾と prev は一致してる（prev が直前に足した画像）前提で、
        # result の末尾を差し替えるイメージで更新する。
        # 一番簡単にやるため、result を「前までの結果」ではなく「常にmerged結果」にして進める。
        # ただし prev は nxt に更新する（次のマッチは直前画像同士が安定）。
        if idx == 2:
            result = merged
        else:
            # idx>=3: result はすでに prev を含むので、prev 部分を「top」として merged を作る必要がある。
            # 手堅く行くため、ここは result と nxt の間で再マッチする（prevではなく result をtopにする）。
            # ただし結果が微妙なら append されるだけなので破綻しにくい。
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

        prev = nxt

    if not cv2.imwrite(out_path, result):
        raise RuntimeError(f"failed to write: {out_path}")


def main():
    out_path = "output.png"
    search_h = 900
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


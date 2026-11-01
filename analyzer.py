"""Rice grain analyzer - core image-processing logic (no UI).

Measures length, width, whiteness and defects of rice grains from a single
phone photo. Setup assumptions:
  * grains lie separated on a plain, uniform, matte background (black paper works well)
  * a round coin of KNOWN diameter is in the same photo (scale reference)
  * optional: a white paper card in the photo (white / exposure reference)
  * photo taken straight from above, diffuse light, no flash glare
"""
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd


class ScaleError(ValueError):
    """Raised when no reference coin is found; carries an image showing what was detected."""
    debug = None


@dataclass
class Params:
    ref_diameter_mm: float = 27.0      # real diameter of the reference coin
    manual_mm_per_px: float = 0.0      # if > 0, skip coin detection and use this scale
    seg_sensitivity: float = 1.0       # <1 = include fainter objects, >1 = stricter
    broken_ratio: float = 0.75         # broken if length < ratio x typical whole-grain length
    yellow_b: float = 18.0             # mean b* above this = yellow/discoloured
    dark_frac: float = 0.04            # fraction of dark pixels above this = dark spots
    chalk_frac: float = 0.15           # fraction of bright pixels above this = chalky patch


@dataclass
class Result:
    table: pd.DataFrame
    annotated: np.ndarray
    mm_per_px: float
    notes: list
    summary: dict


def load_image(data: bytes, max_side: int = 2500) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image file.")
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def _foreground_mask(bgr: np.ndarray, sens: float) -> np.ndarray:
    """Separate objects from the background.

    The background colour is estimated locally (median over a large window), so uneven
    lighting and shadows from the phone do not matter, then pixels that differ clearly
    from it are kept. The threshold comes from the background's own noise level, so a
    strongly coloured coin cannot hide pale grains."""
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    s = 400.0 / max(H, W)
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = int(0.3 * max(small.shape[:2])) // 2 * 2 + 1
    bg_small = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), k)
    bg = cv2.resize(bg_small, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    d = cv2.GaussianBlur(np.linalg.norm(lab - bg, axis=2), (0, 0), 1.2)
    med = float(np.median(d))
    sigma = 1.4826 * float(np.median(np.abs(d - med)))
    t_low = max(med + 3 * sigma, 4.0) * sens
    low = cv2.morphologyEx((d > t_low).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(low)
    min_area = 1e-4 * H * W
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        reg = np.zeros((h, w), np.uint8)
        cv2.drawContours(reg, [c - np.array([x, y])], -1, 255, -1)
        dd = d[y:y + h, x:x + w]
        inside = reg > 0
        peak = float(np.percentile(dd[inside], 90))
        if peak < 3 * sigma + 5:          # too faint everywhere: noise or a smudge
            continue
        keep = inside & (dd > max(t_low, 0.5 * peak))   # edge at half of THIS object's contrast
        out[y:y + h, x:x + w][keep] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def analyze(bgr: np.ndarray, p: Params) -> Result:
    notes = []
    H, W = bgr.shape[:2]
    mask = _foreground_mask(bgr, p.seg_sensitivity)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 1e-4 * H * W]
    if len(cnts) < 5:
        raise ValueError("Too few objects found. Check background, lighting and sensitivity.")

    areas = np.array([cv2.contourArea(c) for c in cnts])
    info = []
    for c, a_ in zip(cnts, areas):
        solidity = a_ / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        (_, _), r_enc = cv2.minEnclosingCircle(c)
        # how well the shape fills its smallest enclosing circle (1.0 = perfect disc);
        # unlike perimeter-based circularity this is not fooled by noisy pixel edges
        disc = a_ / (np.pi * r_enc * r_enc + 1e-9)
        aspect = max(rw, rh) / (min(rw, rh) + 1e-9)
        info.append((disc, solidity, a_ / (rw * rh + 1e-9), aspect))
    # typical single-grain area: median over compact (convex) objects
    solid_areas = [a_ for a_, (_, sol, _, _) in zip(areas, info) if sol >= 0.88]
    ref = float(np.median(solid_areas if len(solid_areas) >= 3 else areas))

    # --- classify objects: coin / white card / touching grains / non-rice / single grain
    coin = card = None
    n_touch = n_other = 0
    grain_cnts, excluded = [], []
    for c, a_, (disc, solidity, extent, aspect) in zip(cnts, areas, info):
        if a_ > 5 * ref and disc > 0.85 and solidity > 0.93 and aspect < 1.2:
            if coin is None or a_ > cv2.contourArea(coin):
                coin = c
        elif a_ > 12 * ref and extent > 0.8:
            card = c
        elif a_ > 12 * ref:
            n_other += 1                 # paper edge, shadow, hand ... not rice
            excluded.append(c)
        elif a_ > 1.5 * ref or solidity < 0.85:
            n_touch += 1                 # two or more grains stuck together: not measured
            excluded.append(c)
        else:
            grain_cnts.append(c)

    if n_touch:
        notes.append(f"{n_touch} touching/overlapping grain group(s) excluded from all results (grey outline).")
    if n_other:
        notes.append(f"{n_other} large non-rice area(s) ignored.")

    # --- scale
    if p.manual_mm_per_px > 0:
        mmpp = p.manual_mm_per_px
        notes.append(f"Using manual scale: {mmpp:.4f} mm/pixel.")
    elif coin is not None:
        d_px = 2 * np.sqrt(cv2.contourArea(coin) / np.pi)
        mmpp = p.ref_diameter_mm / d_px
        notes.append(f"Coin found; scale {mmpp:.4f} mm/pixel.")
    else:
        dbg = bgr.copy()
        cv2.drawContours(dbg, cnts, -1, (0, 0, 255), max(W // 800, 1) + 1)
        err = ScaleError("No reference coin found. It must be a clearly round object, not touching "
                         "grains and fully inside the photo, or enter a manual scale. "
                         "Red outlines show what the app detected.")
        err.debug = dbg
        raise err

    # --- colour normalisation using the white card
    f = bgr.astype(np.float32) / 255.0
    if card is not None:
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [card], -1, 255, -1)
        m = cv2.erode(m, np.ones((15, 15), np.uint8))
        f = np.clip(f / (f[m > 0].mean(axis=0) + 1e-6), 0, 1)
        notes.append("White card found; colours normalised to it.")
    else:
        bgpx = f[mask == 0]
        bgcol = np.median(bgpx, axis=0) if len(bgpx) > 1000 else None
        if bgcol is not None and bgcol.mean() > 0.55:   # light background: assume white paper
            f = np.clip(f / (bgcol + 1e-6), 0, 1)
            notes.append("No white card; the white paper background was used as the white reference.")
        else:
            notes.append("No white card found; whiteness values are relative only.")
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)

    # --- per-grain measurements
    rows, kept = [], []
    for c in grain_cnts:
        x, y, w, h = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue  # touches image edge
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        x0, y0 = max(x - 3, 0), max(y - 3, 0)
        sub = lab[y0:y + h + 3, x0:x + w + 3]
        m = np.zeros(sub.shape[:2], np.uint8)
        cv2.drawContours(m, [c - np.array([x0, y0])], -1, 255, -1)
        n_er = max(2, int(round(0.25 * min(rw, rh))))   # stay away from blurred/shaded edges
        for it in range(n_er, -1, -1):                  # thin grains (low-res photos): erode less
            me = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=it) if it else m
            if int((me > 0).sum()) >= 30:
                break
        px = sub[me > 0]
        if len(px) < 30:
            continue
        L, a, b = px[:, 0], px[:, 1], px[:, 2]
        Lm, am, bm = float(L.mean()), float(a.mean()), float(b.mean())
        medL = float(np.median(L))
        rows.append({
            "length_mm": max(rw, rh) * mmpp,
            "width_mm": min(rw, rh) * mmpp,
            "area_mm2": cv2.contourArea(c) * mmpp ** 2,
            "L": Lm, "a": am, "b": bm,
            "whiteness": 100 - np.sqrt((100 - Lm) ** 2 + am ** 2 + bm ** 2),
            "dark_frac": float((L < medL - 25).mean()),
            "chalk_frac": float((L > medL + 15).mean()),
        })
        kept.append(c)
    if not rows:
        raise ValueError("No complete grains found.")

    df = pd.DataFrame(rows)
    df["l_w_ratio"] = df.length_mm / df.width_mm

    # --- defects
    lens = np.sort(df.length_mm.values)
    ref_len = float(np.median(lens[len(lens) // 2:]))  # typical whole grain
    df["broken"] = df.length_mm < p.broken_ratio * ref_len
    df["yellow"] = df.b > p.yellow_b
    df["dark_spots"] = df.dark_frac > p.dark_frac
    df["chalky"] = df.chalk_frac > p.chalk_frac
    flags = ["broken", "yellow", "dark_spots", "chalky"]
    df["defects"] = df[flags].apply(lambda r: ", ".join(k for k in flags if r[k]) or "none", axis=1)
    df.insert(0, "id", np.arange(1, len(df) + 1))

    # --- annotate
    colors = {"broken": (0, 0, 255), "yellow": (0, 165, 255),
              "dark_spots": (255, 0, 255), "chalky": (255, 255, 0), "none": (0, 200, 0)}
    out = bgr.copy()
    fs = max(W / 2500 * 0.9, 0.4)
    th = max(int(W / 1200), 1)
    for (_, r), c in zip(df.iterrows(), kept):
        col = colors[r.defects.split(", ")[0]]
        cv2.drawContours(out, [c], -1, col, th + 1)
        x, y, _, _ = cv2.boundingRect(c)
        cv2.putText(out, str(int(r.id)), (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
    cv2.drawContours(out, excluded, -1, (150, 150, 150), th)
    if coin is not None:
        cv2.drawContours(out, [coin], -1, (255, 128, 0), th + 1)
    if card is not None:
        cv2.drawContours(out, [card], -1, (255, 128, 0), th + 1)

    n = len(df)
    summary = {
        "grains": n,
        "avg_length_mm": df.length_mm.mean(), "std_length_mm": df.length_mm.std(),
        "avg_width_mm": df.width_mm.mean(), "avg_l_w_ratio": df.l_w_ratio.mean(),
        "avg_whiteness": df.whiteness.mean(),
        "broken_pct": 100 * df.broken.mean(), "yellow_pct": 100 * df.yellow.mean(),
        "dark_pct": 100 * df.dark_spots.mean(), "chalky_pct": 100 * df.chalky.mean(),
        "defective_pct": 100 * (df.defects != "none").mean(),
        "whole_grain_length_mm": ref_len,
        "excluded_touching": n_touch,
    }
    return Result(df.round(3), out, mmpp, notes, summary)

"""Assess source images and create a small, quality-driven set of OCR candidates.

Design principles
-----------------
* Each metric is computed from real pixel data – no hard-coded defaults.
* Processing steps are chosen conditionally; no blind stacking of filters.
* Vietnamese diacritics are fine detail (~2–4 px at typical scan resolution);
  any blur or sharpening is applied conservatively to avoid smearing them.
* The perspective-correction path falls back gracefully when a reliable page
  boundary cannot be found.
* All intermediate images can be saved to ``debug_dir`` for offline inspection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageOps


@dataclass(frozen=True)
class OcrImageCandidate:
    name: str
    image: Any
    source_to_candidate: Any
    preprocessing_steps: tuple[str, ...]


# ---------------------------------------------------------------------------
# Lazy OpenCV / NumPy loader
# ---------------------------------------------------------------------------

def _cv_modules():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OCR image preparation requires OpenCV and NumPy.") from exc
    return cv2, np


# ---------------------------------------------------------------------------
# Image quality metrics (all computed from pixel data)
# ---------------------------------------------------------------------------

def _metrics(image: Any) -> dict[str, float]:
    """Pixel-level measurements used for preprocessing routing.

    None of these values represent OCR accuracy – they describe image
    characteristics that influence which preprocessing steps are helpful.
    """
    cv2, np = _cv_modules()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Sharpness: Laplacian variance (high = sharp edges, low = blurry)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Noise: estimate sigma from low-gradient regions using MAD
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1),
    )
    flat_mask = gradient <= np.percentile(gradient, 35)
    residual = gray.astype("float32") - cv2.GaussianBlur(gray, (0, 0), 1.1).astype("float32")
    samples = residual[flat_mask]
    noise_sigma = (
        1.4826 * float(np.median(np.abs(samples - np.median(samples))))
        if samples.size
        else 0.0
    )

    # Impulse noise: fraction of pixels far from their 3×3 median
    median_filtered = cv2.medianBlur(gray, 3)
    impulse_fraction = float(
        np.mean(np.abs(gray.astype("int16") - median_filtered.astype("int16")) > 35)
    )

    p10, p90 = np.percentile(gray, (10, 90))

    # Additional quality cues
    # Mean HSV colour (hue, saturation, value) – helps detect coloured paper/ink
    color_mean_hsv = np.mean(hsv, axis=(0, 1))
    # Edge density as proportion of edge pixels (Canny) – useful for border detection
    edges_for_density = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges_for_density > 0))
    # Skew reliability proxy – low gradient variance indicates reliable skew estimate
    skew_reliability = 1.0 - (np.std(gradient) / 255.0)

    return {
        "brightness_mean_0_255": round(float(np.mean(gray)), 2),
        "contrast_stddev_0_255": round(float(np.std(gray)), 2),
        "ink_background_separation_0_255": round(float(p90 - p10), 2),
        "sharpness_laplacian_variance": round(laplacian_var, 2),
        "noise_sigma_estimate": round(noise_sigma, 2),
        "impulse_noise_fraction": round(impulse_fraction, 5),
        "color_saturation_mean_0_255": round(float(np.mean(hsv[:, :, 1])), 2),
        "underexposed_fraction": round(float(np.mean(gray < 35)), 4),
        "overexposed_fraction": round(float(np.mean(gray > 235)), 4),
        "color_mean_hsv": tuple(round(float(v), 2) for v in color_mean_hsv),
        "edge_density": round(edge_density, 4),
        "skew_reliability": round(skew_reliability, 3),
    }


def _quality(metrics: dict[str, float]) -> dict[str, Any]:
    """Convert raw metrics into a labelled quality score (0–100)."""
    exposure = max(0.0, 1.0 - abs(metrics["brightness_mean_0_255"] - 145) / 145)
    contrast = min(1.0, metrics["ink_background_separation_0_255"] / 110)
    sharpness = min(1.0, metrics["sharpness_laplacian_variance"] / 250)
    noise = max(0.0, 1.0 - metrics["noise_sigma_estimate"] / 18)
    score = round(100 * (0.28 * exposure + 0.27 * contrast + 0.30 * sharpness + 0.15 * noise), 1)
    return {
        "score_0_100": score,
        "label": "GOOD" if score >= 75 else "FAIR" if score >= 50 else "POOR",
        "components": {
            "exposure": round(exposure, 3),
            "contrast": round(contrast, 3),
            "sharpness": round(sharpness, 3),
            "noise": round(noise, 3),
        },
        "note": "Chỉ số ảnh dùng để chọn tiền xử lý; không phải confidence OCR.",
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _order_corners(points: Any) -> Any:
    """Return corners in [top-left, top-right, bottom-right, bottom-left] order."""
    _cv2, np = _cv_modules()
    points = np.asarray(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]
    ordered[0] = points[np.argmin(sums)]   # top-left
    ordered[2] = points[np.argmax(sums)]   # bottom-right
    ordered[1] = points[np.argmax(diffs)]  # top-right
    ordered[3] = points[np.argmin(diffs)]  # bottom-left
    return ordered


def _find_page(image: Any) -> dict[str, Any] | None:
    """Detect a credible page quadrilateral; return None when not found.

    Improvements over the previous version:
    - Three-level morphological closing to bridge gaps in difficult edges.
    - Convexity check *and* aspect-ratio guard.
    - Confidence formula includes convexity ratio.
    - Area threshold lowered to 0.12 to handle close-up crops where the
      page fills most of the frame.
    """
    cv2, np = _cv_modules()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Two-pass edge detection: first on raw, then on blurred image
    # Using the better result (more closed edges) for contour finding
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 35, 120)

    # Morphological closing to bridge small gaps in document borders.
    # Kernel size scales with image size.
    k = max(5, min(19, int(min(width, height) / 120) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(width * height)
    best: dict[str, Any] | None = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
        area = float(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        # Minimum area: 12 % of frame (lowered from 18 % for close-up shots)
        if area < image_area * 0.12 or perimeter <= 0:
            continue

        # Try progressively coarser approximation until we get exactly 4 corners
        polygon: Any = None
        for epsilon in (0.010, 0.018, 0.028, 0.040):
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approx) == 4:
                polygon = approx
                break

        if polygon is None or not cv2.isContourConvex(polygon):
            continue

        corners = _order_corners(polygon)
        tl, tr, br, bl = corners
        top = float(np.linalg.norm(tr - tl))
        bottom = float(np.linalg.norm(br - bl))
        left_side = float(np.linalg.norm(bl - tl))
        right_side = float(np.linalg.norm(br - tr))

        # Reject degenerate shapes
        if min(top, bottom, left_side, right_side) < 32:
            continue
        # Aspect ratio must be within reasonable document proportions
        max_w = max(top, bottom)
        max_h = max(left_side, right_side)
        if not (0.20 <= max_w / max(1.0, max_h) <= 5.0):
            continue

        # Calculate page orientation angle and aspect ratio
        top_vec = tr - tl
        top_angle = float(np.degrees(np.arctan2(top_vec[1], top_vec[0])))
        bot_vec = br - bl
        bot_angle = float(np.degrees(np.arctan2(bot_vec[1], bot_vec[0])))
        page_angle = round(float((top_angle + bot_angle) / 2.0), 2)
        page_aspect = round(float(max_w / max(1.0, max_h)), 3)

        area_ratio = area / image_area
        # Perspective distortion: ratio of opposite side lengths
        opposite_ratio = max(
            max(top, bottom) / max(1.0, min(top, bottom)),
            max_h / max(1.0, min(left_side, right_side)),
        )
        distortion = min(1.0, (opposite_ratio - 1.0) / 0.35)

        # Convexity: how close the polygon area is to its convex hull area
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        convexity = area / max(1.0, hull_area)

        confidence = (
            min(1.0, area_ratio / 0.75)
            * convexity
            * 0.92
        )
        candidate: dict[str, Any] = {
            "corners": corners,
            "confidence": round(float(confidence), 3),
            "area_ratio": round(area_ratio, 3),
            "perspective_distortion_score": round(float(distortion), 3),
            "page_angle_degrees": page_angle,
            "page_aspect_ratio": page_aspect,
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate

    return best


def _estimate_skew(image: Any) -> dict[str, float | int | None]:
    """Estimate text skew angle using Hough line detection.

    Returns the weighted-median angle in degrees, the number of
    contributing line segments, and the IQR dispersion.
    A high dispersion indicates the image may have multiple text
    orientations or the estimate is unreliable.
    """
    cv2, np = _cv_modules()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 130)

    minimum_length = max(35, int(width * 0.12))
    lines = cv2.HoughLinesP(
        edges,
        1,
        float(np.pi) / 720,
        max(20, min(width, height) // 30),
        minLineLength=minimum_length,
        maxLineGap=max(8, width // 90),
    )
    angles: list[tuple[float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = float(np.hypot(x2 - x1, y2 - y1))
            # Accept angles up to ±30° – handles moderately tilted scans
            if length >= minimum_length and abs(angle) <= 30:
                angles.append((angle, length))

    if not angles:
        return {"degrees": None, "line_count": 0, "dispersion_degrees": None}

    angles.sort(key=lambda item: item[0])
    total_length = sum(length for _a, length in angles)
    halfway, accumulated = total_length / 2, 0.0
    result = angles[-1][0]
    for angle, length in angles:
        accumulated += length
        if accumulated >= halfway:
            result = angle
            break

    values = np.asarray([a for a, _l in angles])
    dispersion = float(np.percentile(values, 75) - np.percentile(values, 25))
    return {
        "degrees": round(result, 2),
        "line_count": len(angles),
        "dispersion_degrees": round(dispersion, 2),
    }


def _rotate(image: Any, angle: float) -> tuple[Any, Any]:
    """Rotate image by *angle* degrees, expanding canvas to avoid clipping."""
    cv2, np = _cv_modules()
    h, w = image.shape[:2]
    affine = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(affine[0, 0]), abs(affine[0, 1])
    new_w = max(1, int(h * sin_a + w * cos_a))
    new_h = max(1, int(h * cos_a + w * sin_a))
    affine[0, 2] += new_w / 2 - w / 2
    affine[1, 2] += new_h / 2 - h / 2
    rotated = cv2.warpAffine(
        image, affine, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    full_matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype("float64")
    return rotated, full_matrix


def _resize(
    image: Any,
    transform: Any,
    target_short_edge: int,
    max_dimension: int,
) -> tuple[Any, Any, bool]:
    """Upscale small images and downscale very large ones within safe bounds."""
    cv2, np = _cv_modules()
    h, w = image.shape[:2]
    short = min(h, w)
    scale = min(2.0, target_short_edge / short) if short < target_short_edge else 1.0
    if max(h, w) * scale > max_dimension:
        scale = max_dimension / max(h, w)
    if 0.97 <= scale <= 1.03:
        return image, transform, False
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    output = cv2.resize(image, size, interpolation=interpolation)
    scale_matrix = np.asarray(
        [[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype="float64"
    )
    return output, scale_matrix @ transform, True


# ---------------------------------------------------------------------------
# Enhancement pipeline (each function is independent and purely additive)
# ---------------------------------------------------------------------------

def _gamma(image: Any, gamma: float) -> Any:
    """Apply gamma correction via LUT."""
    cv2, np = _cv_modules()
    table = np.asarray(
        [((v / 255.0) ** gamma) * 255 for v in range(256)], dtype="uint8"
    )
    return cv2.LUT(image, table)


def _enhance(image: Any, metrics: dict[str, float]) -> tuple[Any, list[str]]:
    """Colour-aware enhancement:
    - If document is coloured (saturation >= 28): apply CLAHE on Value channel (HSV)
      to enhance contrast without causing colour distortion.
    - If grayscale-like (saturation < 28): apply CLAHE on Luminance (LAB L channel).
    - Gamma correction when underexposed (< 70) or overexposed (> 215).
    """
    cv2, _np = _cv_modules()
    output, steps = image.copy(), []

    brightness = metrics.get("brightness_mean_0_255", 128.0)
    if brightness < 70:
        output = _gamma(output, 0.65)
        steps.append("gamma_brightening")
    elif brightness > 215:
        output = _gamma(output, 1.30)
        steps.append("gamma_darkening")

    saturation = metrics.get("color_saturation_mean_0_255", 0.0)
    if saturation >= 28:
        # Coloured document -> CLAHE on HSV Value channel
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_ch = clahe.apply(v_ch)
        output = cv2.cvtColor(cv2.merge((h_ch, s_ch, v_ch)), cv2.COLOR_HSV2BGR)
        steps.append("clahe_hsv_value_channel")
    else:
        # Grayscale / monochrome document -> CLAHE on LAB L channel
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        output = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        steps.append("clahe_luminance")
    return output, steps


def _denoise(image: Any, metrics: dict[str, float]) -> tuple[Any, str] | None:
    """Adaptive denoise selection based on a composite score:
    impulse (weight 2), gaussian sigma (weight 1), sharpness deficit (weight 0.5).
    Lower score → less aggressive denoising.
    """
    cv2, _np = _cv_modules()
    impulse = metrics.get("impulse_noise_fraction", 0.0)
    sigma = metrics.get("noise_sigma_estimate", 0.0)
    sharpness = metrics.get("sharpness_laplacian_variance", 150.0)

    impulse_score = min(1.0, impulse * 2.0)
    sigma_score = min(1.0, sigma / 20.0)
    sharp_deficit = max(0.0, 1.0 - sharpness / 250.0)

    composite = (impulse_score * 2.0 + sigma_score + sharp_deficit * 0.5) / 3.5

    if composite < 0.10:
        return None
    if impulse >= 0.010 and impulse_score >= sigma_score:
        return cv2.medianBlur(image, 3), "median_blur_for_impulse_noise"
    if sigma >= 14 and min(image.shape[:2]) >= 500:
        return (
            cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21),
            "non_local_means_for_high_noise",
        )
    return (
        cv2.bilateralFilter(image, 7, 30, 30),
        "bilateral_filter_for_edge_preserving_denoise",
    )


def _sharpen(image: Any, metrics: dict[str, float]) -> tuple[Any, str] | None:
    """Diacritic-safe sharpening: apply unsharp mask only on high-frequency regions."""
    cv2, _np = _cv_modules()
    sharpness = metrics.get("sharpness_laplacian_variance", 150.0)
    noise = metrics.get("noise_sigma_estimate", 0.0)

    if sharpness >= 80 or noise >= 10:
        return None

    strength = min(0.35, max(0.15, 1.0 - sharpness / 80) * 0.40)
    sigma = 1.2 if sharpness < 30 else 0.8
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    unsharp = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    thresh = 0.2 * float(lap.max()) if lap.size else 0.0
    mask = (lap.astype(_np.float32) > thresh).astype(_np.uint8)
    mask_3ch = cv2.merge([mask, mask, mask])

    sharpened = _np.where(mask_3ch > 0, unsharp, image)
    return sharpened, f"unsharp_mask_diagonal_strength_{strength:.2f}"


def _threshold(image: Any, metrics: dict[str, float]) -> Any:
    """Produce a binary image candidate.

    - Uses Otsu thresholding when the image has low saturation (near-grayscale)
      and good contrast – Otsu gives a globally optimal threshold.
    - Falls back to adaptive Gaussian thresholding otherwise.
    """
    cv2, _np = _cv_modules()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    short_edge = min(gray.shape[:2])

    saturation = metrics.get("color_saturation_mean_0_255", 100)
    separation = metrics.get("ink_background_separation_0_255", 0)

    if saturation < 25 and separation >= 60:
        # Otsu works best on bimodal (ink + paper) histograms
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block = max(15, min(51, (short_edge // 18) | 1))
        block = max(3, (short_edge - 1) | 1) if block >= short_edge else block
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 9
        )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Debug image saving
# ---------------------------------------------------------------------------

def _save_debug(candidates: list[OcrImageCandidate], directory: str | None) -> list[str]:
    """Save candidates and corresponding JSON metric files for visual inspection."""
    if not directory:
        return []
    cv2, _np = _cv_modules()
    folder = Path(directory).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"ocr_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    saved = []
    for candidate in candidates:
        img_path = folder / f"{prefix}_{candidate.name}.png"
        meta_path = folder / f"{prefix}_{candidate.name}_metrics.json"
        if cv2.imwrite(str(img_path), candidate.image):
            saved.append(str(img_path))
            cm = _metrics(candidate.image)
            meta = {
                "candidate_name": candidate.name,
                "preprocessing_steps": list(candidate.preprocessing_steps),
                "width_px": candidate.image.shape[1],
                "height_px": candidate.image.shape[0],
                "metrics": cm,
                "quality": _quality(cm),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
    return saved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_ocr_candidates(
    content: bytes,
    *,
    mode: str = "auto",
    max_candidates: int = 4,
    target_short_edge: int = 1200,
    max_dimension: int = 3500,
    debug_dir: str | None = None,
) -> tuple[list[OcrImageCandidate], dict[str, Any]]:
    """Decode the uploaded image once and generate a bounded set of OCR candidates.

    Each candidate applies a *different* preprocessing strategy so that
    ``ocr_adapter`` can select the one that yields the best OCR result.
    Candidates are ordered from least to most processed, ensuring the
    original image is always tried.

    Parameters
    ----------
    content:
        Raw image bytes (PNG / JPEG / WEBP / BMP supported via OpenCV).
    mode:
        ``"auto"``        – choose strategies based on image quality (default).
        ``"original"``    – only the source image.
        ``"enhanced"``    – always include the enhanced-colour variant.
        ``"threshold"``   – always include the binary variant.
    max_candidates:
        Upper bound on the number of candidates (1–6).
    target_short_edge:
        Upscale so the shorter edge reaches this pixel count (300–2400).
    max_dimension:
        Hard upper bound on either dimension after scaling (800–6000).
    debug_dir:
        If given, all candidate images are saved here as PNG files.
    """
    cv2, np = _cv_modules()
    if mode not in {"auto", "original", "enhanced", "threshold"}:
        mode = "auto"

    # Validate and decode
    exif_corrected = False
    pil_source = None
    try:
        with Image.open(BytesIO(content)) as header:
            # Handle orientation for mobile phone photos (90, 180, 270 degrees)
            transposed = ImageOps.exif_transpose(header)
            if transposed.size != header.size:
                exif_corrected = True
            declared_w, declared_h = transposed.size
            if declared_w * declared_h > 60_000_000:
                raise ValueError("Image exceeds the 60 megapixel processing limit.")
            # Convert to RGB numpy array for guaranteed decoding (TIFF, PNG, JPEG, etc.)
            rgb_conv = transposed.convert("RGB")
            pil_source = np.array(rgb_conv)[:, :, ::-1].copy()  # RGB to BGR
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The uploaded image cannot be decoded.") from exc

    # Attempt OpenCV imdecode; fallback to PIL decoded array if OpenCV fails on TIFF/custom formats
    source = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None or exif_corrected:
        source = pil_source

    if source is None:
        raise ValueError("The uploaded image cannot be decoded.")

    max_candidates = max(1, min(6, max_candidates))
    source_metrics = _metrics(source)
    source_skew = _estimate_skew(source)
    page = _find_page(source)
    warnings: list[str] = []
    candidates: list[OcrImageCandidate] = []

    def add(name: str, image: Any, transform: Any, steps: list[str]) -> None:
        if len(candidates) >= max_candidates or any(c.name == name for c in candidates):
            return
        resized, mapped, changed = _resize(image, transform, target_short_edge, max_dimension)
        all_steps = [*steps, *( ["resize_for_ocr"] if changed else [])]
        candidates.append(OcrImageCandidate(name, resized, mapped, tuple(all_steps)))

    identity = np.eye(3, dtype="float64")

    # --- Always include the original (possibly resized) ---
    add("original", source, identity, [])

    # --- Geometry correction (perspective + deskew) ---
    geometry, transform, geo_steps = source, identity, []
    perspective_applied = False

    if page and page["confidence"] >= 0.50 and page["perspective_distortion_score"] >= 0.03:
        if page["perspective_distortion_score"] > 0.28:
            warnings.append(
                "Độ biến dạng phối cảnh quá lớn (> 0.28); "
                "chuyển sang deskew fallback an toàn thay vì warpPerspective để tránh kéo dãn hoặc méo chữ."
            )
        else:
            tl, tr, br, bl = page["corners"]
            dst_w = max(32, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
            dst_h = max(32, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
            dst_pts = np.asarray(
                [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                dtype="float32",
            )
            matrix = cv2.getPerspectiveTransform(page["corners"], dst_pts).astype("float64")
            geometry = cv2.warpPerspective(
                source, matrix, (dst_w, dst_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            transform, geo_steps = matrix, ["perspective_correction"]
            perspective_applied = True
    elif page is None:
        warnings.append(
            "Không phát hiện được biên trang đủ tin cậy; "
            "giữ ảnh gốc và chỉ đánh giá deskew."
        )
    elif page["confidence"] < 0.50:
        warnings.append(
            "Biên trang có độ tin cậy thấp; "
            "không áp dụng phối cảnh để tránh cắt mép nội dung."
        )

    # Deskew after perspective correction
    geometry_skew = _estimate_skew(geometry)
    angle = geometry_skew["degrees"]
    dispersion = geometry_skew["dispersion_degrees"]

    # Apply deskew for angles 0.4°–20° with low dispersion (reliable estimate)
    deskew_applied = bool(
        angle is not None
        and 0.4 <= abs(float(angle)) <= 20
        and (dispersion or 99) <= 5.5
    )
    if deskew_applied:
        geometry, rotation = _rotate(geometry, float(angle))
        transform = rotation @ transform
        geo_steps = [*geo_steps, "deskew"]
    elif angle is not None and abs(float(angle)) > 20:
        warnings.append(
            "Góc nghiêng ước tính quá lớn (> 20°); "
            "bỏ qua deskew tự động để tránh làm hỏng bố cục."
        )

    if geo_steps:
        add("geometry_corrected", geometry, transform, geo_steps)

    # --- Enhancement decisions based on geometry-corrected image metrics ---
    base_metrics = _metrics(geometry)
    low_contrast = (
        base_metrics["ink_background_separation_0_255"] < 70
        or base_metrics["brightness_mean_0_255"] < 80
        or base_metrics["brightness_mean_0_255"] > 210
    )
    grayscale_like = base_metrics["color_saturation_mean_0_255"] < 20

    # Denoising (at most one method applied)
    if mode != "original":
        denoised = _denoise(geometry, base_metrics)
        if denoised:
            denoised_image, denoise_method = denoised
            add("denoised", denoised_image, transform, [*geo_steps, denoise_method])

    # Enhanced colour (CLAHE ± gamma)
    if mode in {"auto", "enhanced"} and (low_contrast or mode == "enhanced"):
        enhanced_image, enhance_steps = _enhance(geometry, base_metrics)
        add("enhanced_color", enhanced_image, transform, [*geo_steps, *enhance_steps])

    # Binary / threshold candidate
    use_threshold = mode == "threshold" or (
        mode == "auto" and low_contrast and grayscale_like
    )
    if mode in {"auto", "threshold"} and use_threshold:
        add(
            "adaptive_binary",
            _threshold(geometry, base_metrics),
            transform,
            [*geo_steps, "adaptive_or_otsu_threshold"],
        )

    # Sharpening candidate
    if mode != "original":
        sharpened = _sharpen(geometry, base_metrics)
        if sharpened:
            sharp_image, sharp_step = sharpened
            add("sharpened", sharp_image, transform, [*geo_steps, sharp_step])

    # --- Build diagnostics ---
    primary_metrics = _metrics(candidates[0].image)
    candidate_plan = []
    for item in candidates:
        cm = _metrics(item.image)
        candidate_plan.append({
            "name": item.name,
            "steps": list(item.preprocessing_steps),
            "width_px": item.image.shape[1],
            "height_px": item.image.shape[0],
            "quality": {"metrics": cm, **_quality(cm)},
        })

    diagnostics: dict[str, Any] = {
        "width_px": source.shape[1],
        "height_px": source.shape[0],
        "image_quality_before": {"metrics": source_metrics, **_quality(source_metrics)},
        "image_quality_primary_candidate": {"metrics": primary_metrics, **_quality(primary_metrics)},
        # Flat fields kept for backward compatibility
        "brightness_mean_0_255": source_metrics["brightness_mean_0_255"],
        "contrast_stddev_0_255": source_metrics["contrast_stddev_0_255"],
        "blur_laplacian_variance": source_metrics["sharpness_laplacian_variance"],
        "noise_indicator_median_residual": source_metrics["noise_sigma_estimate"],
        "estimated_skew_degrees": source_skew["degrees"],
        "estimated_skew_before_degrees": source_skew["degrees"],
        "estimated_skew_after_geometry_degrees": geometry_skew["degrees"],
        "skew_line_count": geometry_skew["line_count"],
        "skew_dispersion_degrees": geometry_skew["dispersion_degrees"],
        "document_boundary_detected": page is not None,
        "page_boundary_detected": page is not None,
        "document_boundary_confidence": page["confidence"] if page else None,
        "geometry_confidence": round(page["confidence"] * max(0.1, 1.0 - (dispersion or 0.0) / 25.0), 3) if page else (0.50 if deskew_applied else 0.0),
        "document_boundary_area_ratio": page["area_ratio"] if page else None,
        "page_angle_degrees": page.get("page_angle_degrees") if page else None,
        "page_aspect_ratio": page.get("page_aspect_ratio") if page else None,
        "perspective_distortion_score": page["perspective_distortion_score"] if page else None,
        "perspective_correction_applied": perspective_applied,
        "deskew_applied": deskew_applied,
        "exif_orientation_corrected": exif_corrected,
        "colour_document": bool(source_metrics["color_saturation_mean_0_255"] >= 28),
        "candidate_plan": candidate_plan,
        "debug_image_paths": _save_debug(candidates, debug_dir),
        "warnings": warnings,
    }
    return candidates, diagnostics


def prepare_image_for_ocr(
    content: bytes, mode: str = "auto"
) -> tuple[Any, dict[str, Any], Any]:
    """Compatibility wrapper for callers that need a single prepared image."""
    candidates, diagnostics = prepare_ocr_candidates(content, mode=mode, max_candidates=1)
    return candidates[0].image, diagnostics, candidates[0].source_to_candidate


def map_box_to_source(
    box: Any, source_to_candidate: Any, source_size: tuple[int, int]
) -> list[float] | None:
    """Map an OCR bounding box from candidate-image coordinates back to the source."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    cv2, np = _cv_modules()
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
        inverse = np.linalg.inv(source_to_candidate)
        points = np.asarray([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype="float32")
        mapped = cv2.perspectiveTransform(points, inverse.astype("float32"))[0]
        w, h = source_size
        xs = np.clip(mapped[:, 0], 0, max(0, w - 1))
        ys = np.clip(mapped[:, 1], 0, max(0, h - 1))
        return [
            round(float(xs.min()), 2),
            round(float(ys.min()), 2),
            round(float(xs.max()), 2),
            round(float(ys.max()), 2),
        ]
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None


def map_polygon_to_source(
    polygon: Any, source_to_candidate: Any, source_size: tuple[int, int]
) -> list[list[float]] | None:
    """Map an OCR polygon from candidate-image coordinates back to the source."""
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        return None
    cv2, np = _cv_modules()
    try:
        pts = []
        for pt in polygon:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pts.append([float(pt[0]), float(pt[1])])
        if len(pts) < 3:
            return None
        inverse = np.linalg.inv(source_to_candidate)
        points = np.asarray([pts], dtype="float32")
        mapped = cv2.perspectiveTransform(points, inverse.astype("float32"))[0]
        w, h = source_size
        return [
            [
                round(float(np.clip(p[0], 0, max(0, w - 1))), 2),
                round(float(np.clip(p[1], 0, max(0, h - 1))), 2),
            ]
            for p in mapped
        ]
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None

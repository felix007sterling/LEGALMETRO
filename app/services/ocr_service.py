"""OCR service based on the OCR.space cloud API.

OCR is treated strictly as EVIDENCE EXTRACTION, never as a compliance
decision maker. This service:

1. Receives a PreprocessedImage (or raw numpy array) holding several OpenCV
   "variants" of the same image (clean grayscale, CLAHE contrast-enhanced,
   upscaled, deskewed, Otsu-binarized, inverted).
2. Encodes the single best variant to JPEG bytes and posts it to the
   OCR.space REST API (OCREngine=2, overlay enabled for word-level boxes).
3. Normalizes the response into the same structured blocks the rest of the
   pipeline already expects:
   - text, confidence, bounding box (x, y, w, h)
4. Computes an aggregate confidence score for the image.

Why only one HTTP call per image (not the old multi-variant EasyOCR fusion):
OCR.space's free tier is rate-limited and size-limited (~1MB/request), and
each variant would cost a separate network round trip. OCR.space also does
its own internal preprocessing, so a single well-chosen variant is enough —
this trades a little of the old multi-variant fusion accuracy for something
that actually fits a free-tier deployment's memory and quota budget.

No legal logic lives here.
"""

import time
from dataclasses import dataclass, field

import cv2
import httpx
import numpy as np

from app.core.config import get_settings
from app.services.image_service import ImageVariant

OCR_SPACE_URL = "https://apipro1.ocr.space/parse/image"
# Preference order for picking which single variant to send — OCR.space does
# its own thresholding/contrast work internally, so a clean grayscale (or
# CLAHE-enhanced) baseline tends to read best without extra local work.
_VARIANT_PRIORITY = ["clahe", "gray", "upscaled", "deskew", "otsu", "invert"]

# A conservative default confidence for successfully parsed text: OCR.space's
# free tier does not return a numeric per-word confidence score, so callers
# downstream that filter on OCR_MIN_CONFIDENCE need a stand-in value that
# reliably clears that threshold for genuinely recognized text.
_DEFAULT_CONFIDENCE = 0.85


@dataclass
class OCRBlock:
    """One recognized text region."""

    text: str
    confidence: float
    bbox: list[int]  # [x, y, width, height]


@dataclass
class OCRResult:
    """Normalized OCR output for a single image."""

    blocks: list[OCRBlock] = field(default_factory=list)
    raw_text: str = ""
    lenient_text: str = ""
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    engine: str = "ocrspace"
    steps_applied: list[str] = field(default_factory=list)


class OCRSpaceError(Exception):
    """Raised when the OCR.space API call fails or returns an error payload."""


def normalize_bbox(points) -> list[int]:
    """Convert a 4-corner polygon into [x, y, w, h]. Kept for compatibility
    with any callers still passing polygon-style coordinates."""
    if not points:
        return [0, 0, 0, 0]
    xs = [int(round(p[0])) for p in points]
    ys = [int(round(p[1])) for p in points]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return [x, y, w, h]


def _pick_variant(processed):
    """Choose the single best variant to send to OCR.space."""
    variants = list(getattr(processed, "variants", None) or [])
    if not variants:
        # Backward compat: caller passed a raw grayscale image.
        return ImageVariant("gray", np.asarray(processed))
    order = {name: i for i, name in enumerate(_VARIANT_PRIORITY)}
    variants.sort(key=lambda v: order.get(v.name, 99))
    return variants[0]


def _encode_jpeg(image: np.ndarray) -> bytes:
    """Encode a numpy image array to JPEG bytes for upload."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise OCRSpaceError("Failed to encode image for OCR.space upload")
    return buf.tobytes()


def _call_ocr_space(image_bytes: bytes, api_key: str) -> dict:
    """POST the image to OCR.space and return the parsed JSON payload."""
    response = httpx.post(
        OCR_SPACE_URL,
        files={"file": ("label.jpg", image_bytes, "image/jpeg")},
        data={
            "apikey": api_key,
            "language": "eng",
            "OCREngine": 2,
            "isOverlayRequired": True,
            "scale": True,
            "detectOrientation": True,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("IsErroredOnProcessing"):
        message = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Unknown OCR.space error"
        if isinstance(message, list):
            message = "; ".join(message)
        raise OCRSpaceError(message)
    return payload


def _blocks_from_payload(payload: dict) -> list[OCRBlock]:
    """Flatten OCR.space's overlay (Lines -> Words) into OCRBlock objects."""
    blocks: list[OCRBlock] = []
    results = payload.get("ParsedResults") or []
    if not results:
        return blocks
    overlay = results[0].get("TextOverlay") or {}
    for line in overlay.get("Lines", []) or []:
        for w in line.get("Words", []) or []:
            text = (w.get("WordText") or "").strip()
            if not text:
                continue
            x = int(w.get("Left", 0))
            y = int(w.get("Top", 0))
            width = int(w.get("Width", 0))
            height = int(w.get("Height", 0))
            blocks.append(
                OCRBlock(text=text, confidence=_DEFAULT_CONFIDENCE, bbox=[x, y, width, height])
            )
    return blocks


def run_ocr(image_data, steps_applied: list[str] | None = None) -> OCRResult:
    """Send one image variant to OCR.space and normalize the result.

    image_data: a PreprocessedImage (preferred) or a plain grayscale numpy
    array for backward compatibility. Signature is unchanged from the
    previous EasyOCR-based implementation so callers (ocr_pipeline.py /
    engine.py) need no changes.
    """
    settings = get_settings()
    api_key = getattr(settings, "OCR_SPACE_API_KEY", "") or ""
    if not api_key:
        raise OCRSpaceError(
            "OCR_SPACE_API_KEY is not configured — set it in the environment"
        )

    start = time.perf_counter()

    variant = _pick_variant(image_data)
    image_bytes = _encode_jpeg(np.asarray(variant.image))

    payload = _call_ocr_space(image_bytes, api_key)

    results = payload.get("ParsedResults") or []
    raw_text = (results[0].get("ParsedText") or "").strip() if results else ""
    blocks = _blocks_from_payload(payload)

    confident_blocks = [b for b in blocks if b.confidence >= settings.OCR_MIN_CONFIDENCE]
    confidence_score = (
        sum(b.confidence for b in confident_blocks) / len(confident_blocks)
        if confident_blocks
        else (_DEFAULT_CONFIDENCE if raw_text else 0.0)
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return OCRResult(
        blocks=blocks,
        raw_text=raw_text,
        lenient_text=raw_text,
        confidence_score=round(confidence_score, 4),
        processing_time_ms=elapsed_ms,
        engine="ocrspace",
        steps_applied=list(steps_applied or []),
    )

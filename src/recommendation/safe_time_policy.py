import math

# Fitzpatrick scale multipliers for safe sun exposure time.
# Source: ScanSkinAI (http://www.scanskinai.com/safe-sun-exposure-time)
# Formula: safe_minutes = (200 * multiplier) / (3 * max(1.0, uv_index))
SKIN_TYPE_MULTIPLIER = {
    1: 2.5,   # Type I  - Very fair, always burns, never tans
    2: 3.0,   # Type II - Fair, usually burns, sometimes tans
    3: 4.0,   # Type III - Medium, sometimes burns, always tans
    4: 5.0,   # Type IV - Olive, rarely burns, always tans
    5: 8.0,   # Type V  - Brown, very rarely burns
    6: 15.0,  # Type VI - Dark, never burns
}

# Vietnamese Fitzpatrick labels for UI
SKIN_TYPE_LABELS_VI = {
    1: "Loại I - Rất trắng, luôn cháy nắng",
    2: "Loại II - Trắng, dễ cháy nắng",
    3: "Loại III - Trung bình, đôi khi cháy nắng",
    4: "Loại IV - Olive, hiếm khi cháy nắng",
    5: "Loại V - Nâu, rất hiếm cháy nắng",
    6: "Loại VI - Nâu đậm/đen, không bao giờ cháy",
}


def get_safe_exposure_time(skin_type: int, uv_index: float) -> float:
    """
    Calculate safe exposure time in minutes based on Fitzpatrick skin type and UV index.

    Formula: safe_minutes = (200 * multiplier) / (3 * max(1.0, uv_index))

    Aligned with ScanSkinAI formula at:
    http://www.scanskinai.com/safe-sun-exposure-time

    Args:
        skin_type: Fitzpatrick skin type (1-6).
        uv_index: UV index value (0-12+).

    Returns:
        Safe exposure time in minutes (float).

    Raises:
        ValueError: If skin_type is not in 1-6.
    """
    if skin_type not in SKIN_TYPE_MULTIPLIER:
        raise ValueError(f"Invalid skin type: {skin_type}. Must be 1-6 (Fitzpatrick scale).")

    multiplier = SKIN_TYPE_MULTIPLIER[skin_type]
    effective_uv = max(1.0, uv_index)
    return (200.0 * multiplier) / (3.0 * effective_uv)


def validate_against_standards(skin_type: int, uv_index: float) -> dict:
    """
    Validate computed safe time against expected medical reference ranges.

    Returns a dict with:
      - computed_minutes: result of get_safe_exposure_time()
      - in_expected_range: True if value is medically plausible
      - warning: optional warning message
    """
    minutes = get_safe_exposure_time(skin_type, uv_index)

    # At UV=0 or very low (dawn/dusk), time is effectively unlimited - cap check
    if uv_index < 1:
        return {"computed_minutes": minutes, "in_expected_range": True, "warning": None}

    # Medical plausibility: fair skin at UV=11 -> ~1.4 min, dark skin at UV=1 -> 90 min
    in_range = 1.0 <= minutes <= 120.0
    warning = None
    if not in_range:
        warning = (
            f"Unusual safe time {minutes:.1f} min for skin_type={skin_type}, UV={uv_index}. "
            "Check input values."
        )
    return {
        "computed_minutes": round(minutes, 2),
        "in_expected_range": in_range,
        "warning": warning,
    }

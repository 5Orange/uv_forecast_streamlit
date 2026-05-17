import math

# Fitzpatrick scale Minimal Erythemal Dose (MED) values in J/m^2.
# Source: WHO (2002) Global Solar UV Index: A Practical Guide & CIE Action Spectrum
# Formula: safe_minutes = MED_Value / (UV_Index * 1.5)
# (1 UV Index unit = 0.025 W/m^2 = 1.5 J/(m^2*min) of erythemal irradiance)
MED_VALUES_JM2 = {
    1: 200.0,   # Type I  - Very fair, always burns, never tans
    2: 250.0,   # Type II - Fair, usually burns, sometimes tans
    3: 300.0,   # Type III - Medium, sometimes burns, always tans
    4: 450.0,   # Type IV - Olive, rarely burns, always tans
    5: 600.0,   # Type V  - Brown, very rarely burns
    6: 1000.0,  # Type VI - Dark, never burns
}
SKIN_TYPE_MULTIPLIER = {
    1: 2.5,
    2: 3.0,
    3: 4.0,
    4: 5.0,
    5: 8.0,
    6: 15.0,
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

    Formula: safe_minutes = MED_Value / (UV_Index * 1.5)

    Aligned with CIE (1987) Erythemal Action Spectrum and WHO (2002) MED standards.

    Args:
        skin_type: Fitzpatrick skin type (1-6).
        uv_index: UV index value (0-12+).

    Returns:
        Safe exposure time in minutes (float).

    Raises:
        ValueError: If skin_type is not in 1-6.
    """
    if skin_type not in MED_VALUES_JM2:
        raise ValueError(f"Invalid skin type: {skin_type}. Must be 1-6 (Fitzpatrick scale).")

    med_value = MED_VALUES_JM2[skin_type]
    # Cap at 480 min (8h) for negligible UV (dawn/dusk/indoor attenuation)
    if uv_index <= 0.01:
        return 480.0
    # 1 UV Index = 1.5 J/(m^2 * min)
    return min(480.0, med_value / (uv_index * 1.5))


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

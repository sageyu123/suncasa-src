"""Deterministic support-mask and residual QA for EOVSA s00 imaging.

This module intentionally contains no CASA, SunPy, machine-learning, or
pipeline state.  Its functions operate on image-plane arrays so the s00
decision policy remains reproducible and unit-testable.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


PASS_BASE_MASK = "PASS_BASE_MASK"
PASS_EXPANDED_MASK = "PASS_EXPANDED_MASK"
REVIEW_PSF_RESIDUAL = "REVIEW_PSF_RESIDUAL"
FAIL_COVERAGE = "FAIL_COVERAGE"
FAIL_MOTION = "FAIL_MOTION"
FAIL_PSF = "FAIL_PSF"
FAIL_AMBIGUOUS_SOURCE = "FAIL_AMBIGUOUS_SOURCE"


def central_time_indices(time_mjd, center_mjd, duration_hours,
                         min_coverage_fraction=0.90, max_gap_factor=5.0):
    """Select one central, end-exclusive visibility interval.

    :param time_mjd: Monotonic visibility timestamps in MJD.
    :type time_mjd: array-like
    :param center_mjd: Desired central epoch in MJD.
    :type center_mjd: float
    :param duration_hours: Requested full duration in hours.
    :type duration_hours: float
    :param min_coverage_fraction: Minimum sampled/expected timestamp fraction.
    :type min_coverage_fraction: float
    :param max_gap_factor: Largest accepted internal gap in nominal cadences.
    :type max_gap_factor: float
    :returns: Selection metrics including end-exclusive ``interval`` and the
        selected integer ``indices``.
    :rtype: dict
    """
    mjd = np.asarray(time_mjd, dtype=float)
    result = {
        "valid": False,
        "duration_hours": float(duration_hours),
        "interval": None,
        "indices": np.array([], dtype=int),
        "coverage_fraction": 0.0,
        "max_gap_factor": np.inf,
        "reason": "no_samples",
    }
    if mjd.size < 2 or np.any(~np.isfinite(mjd)) or np.any(np.diff(mjd) < 0):
        result["reason"] = "time_grid_not_monotonic"
        return result
    half_days = 0.5 * float(duration_hours) / 24.0
    selected = np.flatnonzero(
        (mjd >= float(center_mjd) - half_days) &
        (mjd <= float(center_mjd) + half_days))
    result["indices"] = selected
    if selected.size < 2:
        return result
    all_steps_sec = np.diff(mjd) * 86400.0
    finite_steps = all_steps_sec[np.isfinite(all_steps_sec) & (all_steps_sec > 0)]
    if finite_steps.size == 0:
        result["reason"] = "invalid_time_cadence"
        return result
    nominal_step_sec = float(np.nanmedian(finite_steps))
    expected_samples = max(2.0, float(duration_hours) * 3600.0 / nominal_step_sec + 1.0)
    coverage = min(1.0, float(selected.size) / expected_samples)
    selected_steps = np.diff(mjd[selected]) * 86400.0
    gap_factor = (float(np.nanmax(selected_steps)) / nominal_step_sec
                  if selected_steps.size else 1.0)
    result.update({
        "interval": [int(selected[0]), int(selected[-1]) + 1],
        "coverage_fraction": coverage,
        "max_gap_factor": gap_factor,
    })
    if coverage < float(min_coverage_fraction):
        result["reason"] = "insufficient_coverage"
    elif gap_factor > float(max_gap_factor):
        result["reason"] = "internal_time_gap"
    else:
        result["valid"] = True
        result["reason"] = "pass"
    return result


def _image2d(data):
    """Return an input image as a finite two-dimensional floating array."""
    image = np.asarray(data, dtype=float).squeeze()
    if image.ndim != 2:
        raise ValueError("image data must reduce to two dimensions")
    return np.where(np.isfinite(image), image, 0.0)


def robust_sigma(values):
    """Estimate Gaussian sigma with a median absolute deviation fallback.

    :param values: Pixel values used for the noise estimate.
    :type values: array-like
    :returns: Non-negative robust noise estimate.
    :rtype: float
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    center = float(np.nanmedian(arr))
    sigma = float(1.4826 * np.nanmedian(np.abs(arr - center)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.sqrt(np.nanmean((arr - center) ** 2)))
    return sigma if np.isfinite(sigma) and sigma > 0 else 0.0


def _disk_structure(radius_pix):
    """Return a circular binary morphology element."""
    radius = max(0, int(np.ceil(float(radius_pix))))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def measure_psf_sidelobe(psf, beam_fwhm_pix, exclusion_fwhm=2.0):
    """Measure the largest absolute PSF sidelobe outside the main lobe.

    :param psf: Point-spread-function image.
    :type psf: array-like
    :param beam_fwhm_pix: Restoring-beam FWHM in pixels.
    :type beam_fwhm_pix: float
    :param exclusion_fwhm: Radius excluded around the PSF peak, in FWHM.
    :type exclusion_fwhm: float
    :returns: Absolute outside peak divided by the absolute PSF peak.
    :rtype: float
    """
    image = _image2d(psf)
    peak_index = np.unravel_index(int(np.nanargmax(np.abs(image))), image.shape)
    peak = float(np.abs(image[peak_index]))
    if peak <= 0:
        return np.inf
    y, x = np.indices(image.shape)
    radius = np.hypot(x - peak_index[1], y - peak_index[0])
    outside = radius > float(exclusion_fwhm) * float(beam_fwhm_pix)
    if not np.any(outside):
        return np.inf
    return float(np.nanmax(np.abs(image[outside])) / peak)


def build_support_mask(model, beam_fwhm_pix, motion_pix=0.0,
                       threshold_sigma=7.0, peak_fraction=0.02,
                       dilation_beams=3.0, clip_mask=None):
    """Build a conservative compact-source support mask from a shallow model.

    :param model: Positive shallow-CLEAN model or trusted init image.
    :type model: array-like
    :param beam_fwhm_pix: Restoring-beam FWHM in pixels.
    :type beam_fwhm_pix: float
    :param motion_pix: Full-window source displacement in pixels.
    :type motion_pix: float
    :param threshold_sigma: Robust seed threshold above the model median.
    :type threshold_sigma: float
    :param peak_fraction: Minimum seed level relative to the positive peak.
    :type peak_fraction: float
    :param dilation_beams: Minimum support dilation radius in beams.
    :type dilation_beams: float
    :param clip_mask: Optional broad solar-disk mask.
    :type clip_mask: array-like or None
    :returns: ``(mask, metrics)`` with a boolean mask and audit dictionary.
    :rtype: tuple(numpy.ndarray, dict)
    :raises ValueError: If no positive source support is present.
    """
    image = _image2d(model)
    finite = np.isfinite(image)
    positive = image[finite & (image > 0)]
    if positive.size == 0:
        raise ValueError("model contains no positive source support")
    median = float(np.nanmedian(image[finite]))
    sigma = robust_sigma(image[finite])
    peak = float(np.nanmax(positive))
    threshold = max(float(peak_fraction) * peak,
                    median + float(threshold_sigma) * sigma if sigma > 0 else 0.0)
    seed = finite & (image >= threshold)
    if not np.any(seed):
        raise ValueError("model threshold produced no support")

    dilation_pix = max(float(dilation_beams) * float(beam_fwhm_pix),
                       float(beam_fwhm_pix) + 0.5 * float(motion_pix))
    mask = ndimage.binary_dilation(seed, structure=_disk_structure(dilation_pix))
    if clip_mask is not None:
        clip = np.asarray(clip_mask, dtype=bool).squeeze()
        if clip.shape != mask.shape:
            raise ValueError("clip mask shape does not match model")
        mask &= clip
    if not np.any(mask):
        raise ValueError("support mask is empty after clipping")
    return mask, {
        "threshold": threshold,
        "noise_sigma": sigma,
        "peak": peak,
        "seed_pixels": int(np.count_nonzero(seed)),
        "mask_pixels": int(np.count_nonzero(mask)),
        "dilation_pix": float(dilation_pix),
    }


def _matched_snr_image(residual, psf, sample_mask):
    """Return a positive matched-filter image, its sigma, and SNR image."""
    image = _image2d(residual)
    kernel = _centered_psf(psf)

    norm = float(np.sqrt(np.sum(kernel * kernel)))
    if norm <= 0:
        filtered = image.copy()
    else:
        filtered = fftconvolve(image, kernel[::-1, ::-1], mode="same") / norm
    sigma = robust_sigma(filtered[np.asarray(sample_mask, dtype=bool)])
    snr = filtered / sigma if sigma > 0 else np.zeros_like(filtered)
    return filtered, sigma, snr


def _centered_psf(psf):
    """Return a PSF whose absolute peak is at the image center."""
    kernel = _image2d(psf)
    peak_index = np.unravel_index(int(np.argmax(np.abs(kernel))), kernel.shape)
    kernel = np.roll(kernel, kernel.shape[0] // 2 - peak_index[0], axis=0)
    kernel = np.roll(kernel, kernel.shape[1] // 2 - peak_index[1], axis=1)
    return kernel


def _predicted_positive_leakage(source_model, support_mask, psf):
    """Predict positive dirty-beam leakage from the in-mask source model."""
    model = _image2d(source_model)
    mask = np.asarray(support_mask, dtype=bool).squeeze()
    if model.shape != mask.shape:
        raise ValueError("source model and support mask shapes do not match")
    prediction = fftconvolve(
        np.where(mask, np.clip(model, 0.0, None), 0.0),
        _centered_psf(psf), mode="same")
    prediction[mask] = 0.0
    return np.clip(prediction, 0.0, None)


def detect_residual_candidates(residual, support_mask, psf, beam_fwhm_pix,
                               source_model=None, in_mask_source_peak=None,
                               snr_trigger=7.0, peak_ratio_trigger=0.10,
                               min_area_beams=0.05):
    """Detect significant positive residual islands outside a support mask.

    A detection only triggers held-out tests; it never authorizes mask growth.

    :param residual: Baseline masked-CLEAN residual image.
    :type residual: array-like
    :param support_mask: Original compact-source support mask.
    :type support_mask: array-like
    :param psf: PSF associated with ``residual``.
    :type psf: array-like
    :param beam_fwhm_pix: Restoring-beam FWHM in pixels.
    :type beam_fwhm_pix: float
    :param source_model: Optional in-mask CLEAN model for PSF-leakage scoring.
    :type source_model: array-like or None
    :param in_mask_source_peak: Optional reference source peak.
    :type in_mask_source_peak: float or None
    :param snr_trigger: Full-window matched-filter trigger threshold.
    :type snr_trigger: float
    :param peak_ratio_trigger: Direct-peak trigger relative to source peak.
    :type peak_ratio_trigger: float
    :param min_area_beams: Minimum candidate area in approximate beam areas.
    :type min_area_beams: float
    :returns: Audit dictionary containing ``triggered`` and candidate records.
    :rtype: dict
    """
    image = _image2d(residual)
    mask = np.asarray(support_mask, dtype=bool).squeeze()
    if mask.shape != image.shape:
        raise ValueError("support mask shape does not match residual")
    outside = ~mask
    filtered, filtered_sigma, snr = _matched_snr_image(image, psf, outside)
    direct_sigma = robust_sigma(image[outside])
    if in_mask_source_peak is None:
        in_mask_source_peak = float(np.nanmax(np.abs(image[mask]))) if np.any(mask) else 0.0
    ratio_level = float(peak_ratio_trigger) * max(float(in_mask_source_peak), 0.0)
    triggered_pixels = outside & (
        (snr >= float(snr_trigger)) |
        ((ratio_level > 0) & (image >= ratio_level))
    )
    labels, nlabels = ndimage.label(triggered_pixels)
    beam_area_pix = np.pi * float(beam_fwhm_pix) ** 2 / (4.0 * np.log(2.0))
    min_pixels = max(1, int(np.ceil(float(min_area_beams) * beam_area_pix)))

    leakage = None
    if source_model is not None:
        model = _image2d(source_model)
        if model.shape == image.shape:
            leakage = _predicted_positive_leakage(model, mask, psf)

    candidates = []
    for label_id in range(1, nlabels + 1):
        island = labels == label_id
        if np.count_nonzero(island) < min_pixels:
            continue
        weights = np.where(island, np.clip(filtered, 0.0, None), 0.0)
        if not np.any(weights > 0):
            continue
        y, x = ndimage.center_of_mass(weights)
        yi = int(np.clip(round(float(y)), 0, image.shape[0] - 1))
        xi = int(np.clip(round(float(x)), 0, image.shape[1] - 1))
        direct_peak = float(np.nanmax(image[island]))
        predicted_leakage = (float(np.nanmax(leakage[island]))
                             if leakage is not None else 0.0)
        candidate = {
            "x": float(x),
            "y": float(y),
            "x_index": xi,
            "y_index": yi,
            "pixels": int(np.count_nonzero(island)),
            "matched_snr": float(np.nanmax(snr[island])),
            "direct_peak": direct_peak,
            "peak_ratio": (direct_peak / float(in_mask_source_peak)
                           if float(in_mask_source_peak) > 0 else np.inf),
            "predicted_psf_leakage": predicted_leakage,
            "psf_explained_fraction": (predicted_leakage / direct_peak
                                       if direct_peak > 0 else np.inf),
        }
        candidates.append(candidate)
    candidates.sort(key=lambda item: item["matched_snr"], reverse=True)
    outside_peak = float(np.nanmax(image[outside])) if np.any(outside) else 0.0
    return {
        "triggered": bool(candidates),
        "candidates": candidates,
        "filtered_sigma": float(filtered_sigma),
        "direct_sigma": float(direct_sigma),
        "outside_peak": outside_peak,
        "outside_peak_ratio": (outside_peak / float(in_mask_source_peak)
                               if float(in_mask_source_peak) > 0 else np.inf),
        "source_peak": float(in_mask_source_peak),
    }


def _local_positive_peak(image, x, y, radius_pix, sample_mask):
    """Measure a local positive peak, SNR, and centroid around one position."""
    data = _image2d(image)
    yy, xx = np.indices(data.shape)
    local = np.hypot(xx - float(x), yy - float(y)) <= float(radius_pix)
    if not np.any(local):
        return {"snr": 0.0, "peak": 0.0, "x": float(x), "y": float(y)}
    sigma = robust_sigma(data[np.asarray(sample_mask, dtype=bool)])
    local_data = np.where(local, data, -np.inf)
    index = np.unravel_index(int(np.nanargmax(local_data)), data.shape)
    peak = float(data[index])
    return {
        "snr": peak / sigma if sigma > 0 else 0.0,
        "peak": peak,
        "x": float(index[1]),
        "y": float(index[0]),
    }


def validate_candidate_repeatability(candidates, time_half_residuals,
                                     support_mask, beam_fwhm_pix,
                                     frequency_residuals=None,
                                     time_half_psfs=None,
                                     frequency_psfs=None,
                                     source_model=None,
                                     required_frequency_count=2,
                                     sigma_min=5.0,
                                     centroid_beam_fraction=0.5,
                                     psf_leakage_veto=0.70):
    """Validate residual candidates on independent time/frequency images.

    :param candidates: Candidate dictionaries from
        :func:`detect_residual_candidates`.
    :type candidates: sequence(dict)
    :param time_half_residuals: Exactly two independently CLEANed residuals.
    :type time_half_residuals: sequence(array-like)
    :param support_mask: Original compact-source support mask.
    :type support_mask: array-like
    :param beam_fwhm_pix: Restoring-beam FWHM in pixels.
    :type beam_fwhm_pix: float
    :param frequency_residuals: Optional independent SPW residuals.
    :type frequency_residuals: sequence(array-like) or None
    :param sigma_min: Required positive SNR in each temporal half.
    :type sigma_min: float
    :param centroid_beam_fraction: Maximum half-to-half centroid separation.
    :type centroid_beam_fraction: float
    :param psf_leakage_veto: Reject candidates explained this strongly by the
        in-mask-source PSF leakage template.
    :type psf_leakage_veto: float
    :returns: Repeatability audit with validated candidate records.
    :rtype: dict
    """
    halves = list(time_half_residuals or [])
    if len(halves) != 2:
        return {"validated": [], "ambiguous": list(candidates), "reason": "missing_time_halves"}
    mask = np.asarray(support_mask, dtype=bool).squeeze()
    search_radius = max(1.0, float(centroid_beam_fraction) * float(beam_fwhm_pix))
    outside = ~mask
    frequency_images = list(frequency_residuals or [])
    half_psfs = list(time_half_psfs or [])
    channel_psfs = list(frequency_psfs or [])
    half_leakage = []
    frequency_leakage = []
    if source_model is not None and len(half_psfs) == len(halves):
        half_leakage = [
            _predicted_positive_leakage(source_model, mask, psf)
            for psf in half_psfs
        ]
    if source_model is not None and len(channel_psfs) == len(frequency_images):
        frequency_leakage = [
            _predicted_positive_leakage(source_model, mask, psf)
            for psf in channel_psfs
        ]
    validated = []
    ambiguous = []
    for candidate in candidates:
        measurements = [
            _local_positive_peak(image, candidate["x"], candidate["y"], search_radius, outside)
            for image in halves
        ]
        separation = float(np.hypot(measurements[0]["x"] - measurements[1]["x"],
                                    measurements[0]["y"] - measurements[1]["y"]))
        time_leakage_fractions = [
            (float(leakage[int(round(item["y"])), int(round(item["x"]))]) / item["peak"]
             if item["peak"] > 0 else np.inf)
            for item, leakage in zip(measurements, half_leakage)
        ]
        time_leakage_ok = (
            len(time_leakage_fractions) == len(measurements)
            and all(value < float(psf_leakage_veto)
                    for value in time_leakage_fractions)
        ) if half_leakage else True
        time_ok = (all(item["snr"] >= float(sigma_min) and item["peak"] > 0
                       for item in measurements)
                   and separation <= search_radius
                   and time_leakage_ok)
        frequency_measurements = [
            _local_positive_peak(image, candidate["x"], candidate["y"], search_radius, outside)
            for image in frequency_images
        ]
        # A full-window residual and two time halves are not independent in
        # frequency.  Require at least one successfully formed SPW diagnostic
        # before a residual island is allowed to grow the CLEAN mask.
        frequency_leakage_fractions = [
            (float(leakage[int(round(item["y"])), int(round(item["x"]))]) / item["peak"]
             if item["peak"] > 0 else np.inf)
            for item, leakage in zip(frequency_measurements, frequency_leakage)
        ]
        frequency_leakage_ok = (
            len(frequency_leakage_fractions) == len(frequency_measurements)
            and all(value < float(psf_leakage_veto)
                    for value in frequency_leakage_fractions)
        ) if frequency_leakage else True
        frequency_ok = (
            len(frequency_measurements) >= int(required_frequency_count)
            and all(item["snr"] >= float(sigma_min) and item["peak"] > 0
                    for item in frequency_measurements)
            and frequency_leakage_ok
        )
        leakage_ok = float(candidate.get(
            "psf_explained_fraction", 0.0)) < float(psf_leakage_veto)
        record = dict(candidate)
        record.update({
            "time_half_measurements": measurements,
            "time_half_separation_pix": separation,
            "frequency_measurements": frequency_measurements,
            "time_psf_explained_fractions": time_leakage_fractions,
            "frequency_psf_explained_fractions": frequency_leakage_fractions,
            "time_ok": bool(time_ok),
            "frequency_ok": bool(frequency_ok),
            "leakage_ok": bool(leakage_ok),
        })
        if time_ok and frequency_ok and leakage_ok:
            record["x"] = float(np.mean([item["x"] for item in measurements]))
            record["y"] = float(np.mean([item["y"] for item in measurements]))
            validated.append(record)
        else:
            ambiguous.append(record)
    reason = "repeatable" if validated else "not_repeatable_or_psf_like"
    return {"validated": validated, "ambiguous": ambiguous, "reason": reason}


def expand_support_mask(support_mask, candidates, beam_fwhm_pix,
                        dilation_beams=1.0, max_growth_fraction=0.10,
                        max_added_beams_per_candidate=3.0,
                        max_candidate_islands=2,
                        max_image_fraction=0.25,
                        clip_mask=None):
    """Add validated local islands while enforcing a strict growth cap.

    :param support_mask: Original compact-source support mask.
    :type support_mask: array-like
    :param candidates: Validated candidate dictionaries with ``x`` and ``y``.
    :type candidates: sequence(dict)
    :param beam_fwhm_pix: Restoring-beam FWHM in pixels.
    :type beam_fwhm_pix: float
    :param dilation_beams: Radius of each local addition in beams.
    :type dilation_beams: float
    :param max_growth_fraction: Maximum added area relative to original mask.
    :type max_growth_fraction: float
    :param clip_mask: Optional broad disk mask.
    :type clip_mask: array-like or None
    :returns: ``(expanded_mask, metrics)``.  ``accepted`` is false when capped.
    :rtype: tuple(numpy.ndarray, dict)
    """
    base = np.asarray(support_mask, dtype=bool).squeeze()
    expanded = base.copy()
    yy, xx = np.indices(base.shape)
    radius = max(1.0, float(dilation_beams) * float(beam_fwhm_pix))
    for candidate in candidates:
        expanded |= np.hypot(xx - float(candidate["x"]), yy - float(candidate["y"])) <= radius
    if clip_mask is not None:
        clip = np.asarray(clip_mask, dtype=bool).squeeze()
        if clip.shape != base.shape:
            raise ValueError("clip mask shape does not match support mask")
        expanded &= clip
    base_pixels = int(np.count_nonzero(base))
    added_pixels = int(np.count_nonzero(expanded & ~base))
    growth_fraction = added_pixels / max(1, base_pixels)
    beam_area_pixels = (np.pi * float(beam_fwhm_pix) ** 2 /
                        (4.0 * np.log(2.0)))
    candidate_count = len(candidates)
    added_pixel_cap = max(
        float(max_growth_fraction) * base_pixels,
        float(max_added_beams_per_candidate) * candidate_count * beam_area_pixels,
    )
    image_fraction = float(np.count_nonzero(expanded)) / float(expanded.size)
    accepted = (
        0 < candidate_count <= int(max_candidate_islands)
        and added_pixels <= added_pixel_cap
        and image_fraction <= float(max_image_fraction)
    )
    return (expanded if accepted else base.copy()), {
        "accepted": accepted,
        "base_pixels": base_pixels,
        "added_pixels": added_pixels,
        "added_pixel_cap": float(added_pixel_cap),
        "beam_area_pixels": float(beam_area_pixels),
        "candidate_count": int(candidate_count),
        "growth_fraction": float(growth_fraction),
        "image_fraction": image_fraction,
        "radius_pix": float(radius),
    }


def validate_retry(before_residual, after_residual, before_model, after_model,
                   base_mask, expanded_mask, candidates, beam_fwhm_pix,
                   residual_drop_min=0.50, energy_drop_min=0.20,
                   rms_worsen_max=0.05, source_flux_change_max=0.10,
                   negative_sigma_max=7.0):
    """Validate one expanded-mask CLEAN against the untouched baseline.

    :returns: Audit dictionary with boolean ``accepted`` and all comparison
        metrics used by the deterministic rollback decision.
    :rtype: dict
    """
    before = _image2d(before_residual)
    after = _image2d(after_residual)
    model0 = _image2d(before_model)
    model1 = _image2d(after_model)
    base = np.asarray(base_mask, dtype=bool).squeeze()
    expanded = np.asarray(expanded_mask, dtype=bool).squeeze()
    if not (before.shape == after.shape == model0.shape == model1.shape == base.shape == expanded.shape):
        raise ValueError("retry images and masks must have identical shapes")

    yy, xx = np.indices(before.shape)
    local = np.zeros(before.shape, dtype=bool)
    radius = max(1.0, float(beam_fwhm_pix))
    for candidate in candidates:
        local |= np.hypot(xx - float(candidate["x"]), yy - float(candidate["y"])) <= radius
    if not np.any(local):
        return {"accepted": False, "reason": "no_retry_candidate"}
    before_peak = float(np.nanmax(np.clip(before[local], 0.0, None)))
    after_peak = float(np.nanmax(np.clip(after[local], 0.0, None)))
    residual_drop = ((before_peak - after_peak) / before_peak if before_peak > 0 else 0.0)
    energy0 = float(np.nansum(before[local] ** 2))
    energy1 = float(np.nansum(after[local] ** 2))
    energy_drop = ((energy0 - energy1) / energy0 if energy0 > 0 else 0.0)
    outside = ~expanded
    rms0 = robust_sigma(before[outside])
    rms1 = robust_sigma(after[outside])
    rms_worsen = ((rms1 - rms0) / rms0 if rms0 > 0 else np.inf)
    flux0 = float(np.nansum(np.clip(model0[base], 0.0, None)))
    flux1 = float(np.nansum(np.clip(model1[base], 0.0, None)))
    source_flux_change = (abs(flux1 - flux0) / flux0 if flux0 > 0 else np.inf)
    negative_sigma = (abs(float(np.nanmin(after[outside]))) / rms1
                      if rms1 > 0 and np.any(outside) else 0.0)
    accepted = (
        residual_drop >= float(residual_drop_min)
        and energy_drop >= float(energy_drop_min)
        and rms_worsen <= float(rms_worsen_max)
        and source_flux_change <= float(source_flux_change_max)
        and negative_sigma <= float(negative_sigma_max)
    )
    return {
        "accepted": bool(accepted),
        "reason": "pass" if accepted else "retry_validation_failed",
        "residual_drop": float(residual_drop),
        "energy_drop": float(energy_drop),
        "rms_worsen": float(rms_worsen),
        "source_flux_change": float(source_flux_change),
        "negative_sigma": float(negative_sigma),
    }


def final_state(coverage_ok=True, motion_ok=True, psf_ok=True, assessment=None,
                repeatability=None, retry=None):
    """Resolve the deterministic s00 QA state.

    :returns: One of the module-level ``PASS_*``, ``REVIEW_*``, or ``FAIL_*``
        constants.
    :rtype: str
    """
    if not coverage_ok:
        return FAIL_COVERAGE
    if not motion_ok:
        return FAIL_MOTION
    if not psf_ok:
        return FAIL_PSF
    assessment = assessment or {}
    if not assessment.get("triggered", False):
        return PASS_BASE_MASK
    repeatability = repeatability or {}
    if not repeatability.get("validated"):
        return REVIEW_PSF_RESIDUAL
    if retry is None:
        return FAIL_AMBIGUOUS_SOURCE
    return PASS_EXPANDED_MASK if retry.get("accepted", False) else FAIL_AMBIGUOUS_SOURCE

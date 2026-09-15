import numpy as np

from suncasa.eovsa import s00_imaging_policy as policy
from suncasa.eovsa.wrap_wsclean import WSClean


def _gaussian(shape=(65, 65), x=32.0, y=32.0, sigma=2.0, amplitude=1.0):
    yy, xx = np.indices(shape)
    return amplitude * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma ** 2)


def test_central_time_indices_are_end_exclusive_and_gap_aware():
    center = 60000.5
    cadence_days = 60.0 / 86400.0
    times = center + np.arange(-180, 181) * cadence_days
    selected = policy.central_time_indices(times, center, duration_hours=5.0)
    assert selected["valid"]
    assert selected["interval"][1] == selected["indices"][-1] + 1
    assert selected["coverage_fraction"] > 0.99

    gapped = np.delete(times, np.arange(170, 190))
    rejected = policy.central_time_indices(
        gapped, center, duration_hours=5.0,
        min_coverage_fraction=0.90, max_gap_factor=5.0)
    assert not rejected["valid"]
    assert rejected["reason"] in {"insufficient_coverage", "internal_time_gap"}


def test_dirty_psf_probe_requests_psf_without_cleaning():
    clean = WSClean("probe.ms")
    clean.setup(niter=0, make_psf=True, quiet=False)
    command = clean.build_command().split()
    assert "-make-psf" in command
    assert "-niter" not in command


def test_measure_psf_sidelobe_excludes_main_lobe():
    psf = _gaussian(sigma=1.2)
    psf[8, 9] = -0.20
    assert np.isclose(policy.measure_psf_sidelobe(psf, 4.0), 0.20, atol=1e-4)


def test_build_support_mask_dilates_and_clips():
    model = np.zeros((65, 65), dtype=float)
    model[32, 32] = 10.0
    yy, xx = np.indices(model.shape)
    clip = np.hypot(xx - 32, yy - 32) <= 10
    mask, metrics = policy.build_support_mask(
        model, beam_fwhm_pix=4.0, dilation_beams=3.0, clip_mask=clip)
    assert mask[32, 32]
    assert np.all(~mask[~clip])
    assert metrics["seed_pixels"] == 1
    assert metrics["mask_pixels"] == int(np.count_nonzero(clip))


def test_residual_candidate_only_triggers_held_out_stage():
    residual = np.zeros((65, 65), dtype=float)
    residual[18, 48] = 2.0
    support = np.zeros_like(residual, dtype=bool)
    support[28:37, 28:37] = True
    psf = np.zeros_like(residual)
    psf[32, 32] = 1.0
    model = np.zeros_like(residual)
    model[32, 32] = 10.0
    result = policy.detect_residual_candidates(
        residual, support, psf, beam_fwhm_pix=4.0,
        source_model=model, in_mask_source_peak=10.0,
        peak_ratio_trigger=0.10)
    assert result["triggered"]
    assert result["candidates"][0]["x_index"] == 48
    assert result["candidates"][0]["y_index"] == 18


def test_residual_candidate_records_amplitude_calibrated_psf_explanation():
    residual = np.zeros((65, 65), dtype=float)
    residual[18, 48] = 2.0
    support = np.zeros_like(residual, dtype=bool)
    support[30:35, 30:35] = True
    model = np.zeros_like(residual)
    model[32, 32] = 10.0
    psf = np.zeros_like(residual)
    psf[32, 32] = 1.0
    psf[18, 48] = 0.2

    result = policy.detect_residual_candidates(
        residual, support, psf, beam_fwhm_pix=4.0,
        source_model=model, in_mask_source_peak=10.0,
        peak_ratio_trigger=0.10)

    candidate = result["candidates"][0]
    assert np.isclose(candidate["predicted_psf_leakage"], 2.0)
    assert np.isclose(candidate["psf_explained_fraction"], 1.0)


def test_repeatability_requires_both_time_halves():
    rng = np.random.default_rng(42)
    support = np.zeros((65, 65), dtype=bool)
    support[28:37, 28:37] = True
    candidate = {"x": 48.0, "y": 18.0, "psf_explained_fraction": 0.1}
    half_a = rng.normal(0.0, 0.02, support.shape)
    half_b = rng.normal(0.0, 0.02, support.shape)
    half_a[18, 48] += 1.0
    half_b[18, 49] += 0.9
    freq = rng.normal(0.0, 0.02, support.shape)
    freq[18, 48] += 0.8
    freq_b = rng.normal(0.0, 0.02, support.shape)
    freq_b[18, 48] += 0.7

    passed = policy.validate_candidate_repeatability(
        [candidate], [half_a, half_b], support, beam_fwhm_pix=4.0,
        frequency_residuals=[freq, freq_b], sigma_min=5.0,
        centroid_beam_fraction=0.5)
    assert len(passed["validated"]) == 1

    half_b[18, 49] = 0.0
    failed = policy.validate_candidate_repeatability(
        [candidate], [half_a, half_b], support, beam_fwhm_pix=4.0,
        frequency_residuals=[freq, freq_b], sigma_min=5.0,
        centroid_beam_fraction=0.5)
    assert not failed["validated"]


def test_repeatability_vetoes_predicted_psf_leakage():
    rng = np.random.default_rng(7)
    support = np.zeros((65, 65), dtype=bool)
    support[28:37, 28:37] = True
    halves = [rng.normal(0.0, 0.01, support.shape) for _ in range(2)]
    for image in halves:
        image[18, 48] += 1.0
    candidate = {"x": 48.0, "y": 18.0, "psf_explained_fraction": 0.9}
    result = policy.validate_candidate_repeatability(
        [candidate], halves, support, beam_fwhm_pix=4.0)
    assert not result["validated"]
    assert not result["ambiguous"][0]["leakage_ok"]


def test_repeatability_requires_independent_frequency_evidence():
    rng = np.random.default_rng(17)
    support = np.zeros((65, 65), dtype=bool)
    support[28:37, 28:37] = True
    halves = [rng.normal(0.0, 0.01, support.shape) for _ in range(2)]
    for image in halves:
        image[18, 48] += 1.0
    candidate = {"x": 48.0, "y": 18.0, "psf_explained_fraction": 0.1}
    result = policy.validate_candidate_repeatability(
        [candidate], halves, support, beam_fwhm_pix=4.0,
        frequency_residuals=[])
    assert not result["validated"]
    assert not result["ambiguous"][0]["frequency_ok"]

    one_frequency = rng.normal(0.0, 0.01, support.shape)
    one_frequency[18, 48] += 1.0
    result = policy.validate_candidate_repeatability(
        [candidate], halves, support, beam_fwhm_pix=4.0,
        frequency_residuals=[one_frequency])
    assert not result["validated"]
    assert not result["ambiguous"][0]["frequency_ok"]


def test_mask_growth_cap_fails_closed():
    yy, xx = np.indices((65, 65))
    base = np.hypot(xx - 32, yy - 32) <= 5
    candidate = {"x": 48.0, "y": 18.0}
    unchanged, rejected = policy.expand_support_mask(
        base, [candidate], beam_fwhm_pix=4.0, max_growth_fraction=0.10,
        max_added_beams_per_candidate=0.0)
    assert not rejected["accepted"]
    assert np.array_equal(unchanged, base)

    expanded, accepted = policy.expand_support_mask(
        base, [candidate], beam_fwhm_pix=4.0, max_growth_fraction=1.0)
    assert accepted["accepted"]
    assert np.count_nonzero(expanded) > np.count_nonzero(base)


def test_mask_growth_allows_one_beam_source_independent_of_small_base_area():
    yy, xx = np.indices((65, 65))
    base = np.hypot(xx - 32, yy - 32) <= 5
    candidate = {"x": 48.0, "y": 18.0}

    expanded, result = policy.expand_support_mask(
        base, [candidate], beam_fwhm_pix=4.0,
        max_growth_fraction=0.01)

    assert result["accepted"]
    assert result["added_pixels"] <= result["added_pixel_cap"]
    assert np.count_nonzero(expanded) > np.count_nonzero(base)


def test_retry_validation_accepts_improvement_without_source_drift():
    rng = np.random.default_rng(11)
    shape = (65, 65)
    yy, xx = np.indices(shape)
    base_mask = np.hypot(xx - 32, yy - 32) <= 6
    expanded_mask = base_mask | (np.hypot(xx - 48, yy - 18) <= 4)
    before = rng.normal(0.0, 0.02, shape)
    after = before.copy()
    before[18, 48] += 2.0
    after[18, 48] += 0.2
    model0 = np.zeros(shape)
    model1 = np.zeros(shape)
    model0[32, 32] = 10.0
    model1[32, 32] = 10.0
    model1[18, 48] = 1.8
    result = policy.validate_retry(
        before, after, model0, model1, base_mask, expanded_mask,
        [{"x": 48.0, "y": 18.0}], beam_fwhm_pix=4.0,
        rms_worsen_max=0.10)
    assert result["accepted"]
    assert result["residual_drop"] > 0.5


def test_final_state_is_fail_closed():
    assert policy.final_state(coverage_ok=False) == policy.FAIL_COVERAGE
    assert policy.final_state(motion_ok=False) == policy.FAIL_MOTION
    assert policy.final_state(psf_ok=False) == policy.FAIL_PSF
    assert policy.final_state(assessment={"triggered": False}) == policy.PASS_BASE_MASK
    assert policy.final_state(
        assessment={"triggered": True}, repeatability={"validated": []}
    ) == policy.REVIEW_PSF_RESIDUAL
    assert policy.final_state(
        assessment={"triggered": True}, repeatability={"validated": [{}]},
        retry={"accepted": True}
    ) == policy.PASS_EXPANDED_MASK

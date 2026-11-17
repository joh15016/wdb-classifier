"""Label-blind eclipse descriptors at cached period hypotheses.

No catalog periods, labels, source coordinates, or other-source measurements
are accepted. Descriptors are computed from the supplied photometry and period
hypotheses, with explicit eclipse-support requirements.
"""

from __future__ import annotations

import numpy as np
from astropy.timeseries import BoxLeastSquares

DUTY_CYCLES = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.20])
MIN_PERIOD = 5 / 1440
MAX_PERIOD = 10.0
OVERSAMPLE = 10
DESCRIPTORS = [
    "depth_relative",
    "depth_snr",
    "duty_cycle",
    "duration_minutes",
    "delta_chi2",
    "n_in",
    "n_nights",
    "n_cycles",
    "phase_coverage",
    "leave_one_out_snr",
    "secondary_snr",
    "secondary_depth_ratio",
    "odd_even_snr",
    "residual_chi2_reduced",
    "box_vs_sinusoid_delta_loglike",
]
SELECTED_FIELDS = [f"eclipse_selected_{name}" for name in DESCRIPTORS]
MULTI_FIELDS = [f"eclipse_best_{name}" for name in DESCRIPTORS] + [
    "eclipse_best_period_days",
    "eclipse_candidate_count",
    "eclipse_supported_fraction",
    "eclipse_nonalias_score_gap",
]


def relative_flux(time, mag, mag_err):
    t, m, e = [np.asarray(a, dtype=np.float64) for a in (time, mag, mag_err)]
    if t.ndim != 1 or t.shape != m.shape or t.shape != e.shape or len(t) < 10:
        raise ValueError("Need at least ten aligned one-dimensional observations")
    if not all(np.isfinite(a).all() for a in (t, m, e)) or (e <= 0).any() or np.ptp(t) <= 0:
        raise ValueError("Invalid epochs, magnitude errors, or time baseline")
    order = np.argsort(t, kind="stable")
    t, m, e = t[order], m[order], e[order]
    flux = 10 ** (-0.4 * (m - np.median(m)))
    error = (0.4 * np.log(10)) * flux * e
    if not np.isfinite(flux).all() or not np.isfinite(error).all() or (error <= 0).any():
        raise ValueError("Invalid magnitude-to-relative-flux transformation")
    # Do not downcast large absolute Julian dates to float32.
    return t - t[0], flux, error, np.floor(t).astype(np.int64)


def period_hypotheses(selected, top):
    """Keep the selected period first, then all cached periods and aliases.

    De-duplicate only essentially identical floats, not frequencies spaced
    closer than 0.1%: that tolerance would merge distinct narrow-eclipse fits.
    """
    if not np.isfinite(selected) or not MIN_PERIOD <= selected <= MAX_PERIOD:
        raise ValueError("Selected period is outside the cached search range")
    values = [float(selected)]
    for algorithm in sorted(top):
        values.extend(float(p) for p in top[algorithm] if np.isfinite(p) and p > 0)
    periods = []
    for factor in (1.0, 0.5, 2.0):
        for period in values:
            p = factor * period
            if MIN_PERIOD <= p <= MAX_PERIOD and not any(
                np.isclose(p, old, rtol=1e-10, atol=0) for old in periods
            ):
                periods.append(p)
    return periods


def eclipse_mask(time, period, epoch, duration):
    offset = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    return np.abs(offset) < 0.5 * duration


def depth_estimate(flux, weight, inside, outside):
    if not inside.any() or not outside.any():
        return np.nan, np.nan
    wi, wo = weight[inside].sum(), weight[outside].sum()
    depth = np.sum(weight[outside] * flux[outside]) / wo - np.sum(weight[inside] * flux[inside]) / wi
    return float(depth), float(np.sqrt(1 / wi + 1 / wo))


def fit_candidate(bls, time, flux, error, nights, period):
    """Fit one candidate using BLS, then summarize observed points only.

    Avoid BLS.compute_stats: at minute periods over years it allocates arrays
    for hundreds of thousands of unobserved orbital cycles. Cycle support
    here counts unique observed cycles instead.
    """
    result = bls.power([period], period * DUTY_CYCLES, objective="snr", method="fast", oversample=OVERSAMPLE)
    epoch, duration = float(result.transit_time[0]), float(result.duration[0])
    inside = eclipse_mask(time, period, epoch, duration)
    outside = ~inside
    weight = 1 / error**2
    depth, depth_err = depth_estimate(flux, weight, inside, outside)
    row = dict(
        period_days=float(period),
        epoch_relative_days=epoch,
        duration_days=duration,
        rank_score=0.0,
        supported=False,
        **{name: np.nan for name in DESCRIPTORS},
    )
    if not np.isfinite(depth_err) or depth <= 0:
        return row
    baseline = np.average(flux[outside], weights=weight[outside])
    model = np.full(len(time), baseline)
    model[inside] = baseline - depth
    chi_box = np.sum(weight * (flux - model) ** 2)
    chi_constant = np.sum(weight * (flux - np.average(flux, weights=weight)) ** 2)
    cycles = np.floor((time - epoch) / period + 0.5).astype(np.int64)
    phase = (time / period) % 1
    secondary = eclipse_mask(time, period, epoch + period / 2, duration)
    out_both = ~(inside | secondary)
    secondary_depth, secondary_error = depth_estimate(flux, weight, secondary, out_both)
    even, odd = inside & (cycles % 2 == 0), inside & (cycles % 2 != 0)
    odd_even_snr = np.nan
    if even.any() and odd.any():
        # Common out-of-eclipse level cancels in the difference.
        difference = np.average(flux[even], weights=weight[even]) - np.average(flux[odd], weights=weight[odd])
        odd_even_snr = abs(difference) / np.sqrt(1 / weight[even].sum() + 1 / weight[odd].sum())
    remaining = inside.copy()
    deepest = np.flatnonzero(inside)[np.argmax((baseline - flux[inside]) / error[inside])]
    remaining[deepest] = False
    # Remove the deepest point entirely, not into the out-of-eclipse baseline.
    loo_depth, loo_error = depth_estimate(flux, weight, remaining, outside)
    loo_snr = loo_depth / loo_error if np.isfinite(loo_error) else 0.0
    angle = 2 * np.pi * phase
    design = np.c_[np.ones(len(time)), np.sin(angle), np.cos(angle)]
    coefficients, *_ = np.linalg.lstsq(design / error[:, None], flux / error, rcond=None)
    chi_sinusoid = np.sum(((flux - design @ coefficients) / error) ** 2)
    n_nights, n_cycles = len(np.unique(nights[inside])), len(np.unique(cycles[inside]))
    row.update(
        depth_relative=float(depth / baseline),
        depth_snr=float(depth / depth_err),
        duty_cycle=float(duration / period),
        duration_minutes=float(duration * 1440),
        delta_chi2=float(max(0, chi_constant - chi_box)),
        n_in=int(inside.sum()),
        n_nights=n_nights,
        n_cycles=n_cycles,
        phase_coverage=float(np.mean(np.histogram(phase, bins=np.linspace(0, 1, 21))[0] > 0)),
        leave_one_out_snr=float(loo_snr),
        secondary_snr=float(secondary_depth / secondary_error),
        secondary_depth_ratio=float(secondary_depth / depth),
        odd_even_snr=float(odd_even_snr),
        residual_chi2_reduced=float(chi_box / max(1, len(time) - 4)),
        box_vs_sinusoid_delta_loglike=float((chi_sinusoid - chi_box) / 2),
    )
    row["supported"] = bool(inside.sum() >= 3 and n_nights >= 2 and n_cycles >= 2 and loo_snr > 0)
    row["rank_score"] = float(max(0, loo_snr)) if row["supported"] else 0.0
    return row


def is_simple_alias(first, second, baseline):
    """Treat nearby/harmonic peaks as one family for the runner-up margin."""
    f = 1 / first
    return any(abs(f - 1 / (second * harmonic)) < 1 / max(baseline, second) for harmonic in (0.5, 1, 2))


def extract(time, mag, mag_err, selected_period, cached_periods):
    t, flux, error, nights = relative_flux(time, mag, mag_err)
    periods = period_hypotheses(selected_period, cached_periods)
    bls = BoxLeastSquares(t, flux, error)
    candidates = [fit_candidate(bls, t, flux, error, nights, p) for p in periods]
    selected = candidates[0]
    # Supported repeated dips outrank unsupported single-point anomalies.
    # On exact ties keep the earlier, label-blind candidate order.
    best_index = max(
        range(len(candidates)),
        key=lambda i: (
            candidates[i]["rank_score"],
            np.nan_to_num(candidates[i]["depth_snr"], nan=-np.inf),
            -i,
        ),
    )
    best = candidates[best_index]
    nonalias = [
        row["rank_score"]
        for row in candidates
        if not is_simple_alias(row["period_days"], best["period_days"], np.ptp(t))
    ]
    values = {f"eclipse_selected_{name}": selected[name] for name in DESCRIPTORS}
    values.update({f"eclipse_best_{name}": best[name] for name in DESCRIPTORS})
    values.update(
        eclipse_best_period_days=best["period_days"],
        eclipse_candidate_count=len(candidates),
        eclipse_supported_fraction=float(np.mean([row["supported"] for row in candidates])),
        eclipse_nonalias_score_gap=float(best["rank_score"] - max(nonalias, default=0)),
    )
    return values, candidates, best_index

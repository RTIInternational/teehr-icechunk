"""Sparse zone-by-pixel weights and coverage-weighted means.

Copied from teehr's ``teehr.fetching.nwm.grid_utils`` (legacy teehr is frozen; ADR-0015).
"""
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse

# Zone key column of a weights table
LOCATION_ID = "location_id"


def _zone_weights_matrix(
    location_ids: np.ndarray,
    pixel_codes: np.ndarray,
    n_pixels: int,
    weights: np.ndarray,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse zone-by-pixel weights matrix, zones sorted by location_id."""
    # Factorize (hash-based) rather than sorting the label column, then sort
    # only the unique labels and remap -- sorting millions of location strings
    # dominates otherwise.
    codes, unique_locations = pd.factorize(location_ids)
    label_order = np.argsort(unique_locations)
    rank = np.empty_like(label_order)
    rank[label_order] = np.arange(label_order.size)
    codes = rank[codes]
    unique_locations = unique_locations[label_order]

    # Accumulate in float64; the weights and values are float32 and a zone can
    # cover thousands of pixels.
    matrix = sparse.csr_matrix(
        (np.asarray(weights, dtype="float64"), (codes, pixel_codes)),
        shape=(unique_locations.size, n_pixels),
    )
    total_weight = matrix @ np.ones(n_pixels)
    if not (total_weight > 0).all():
        empty = unique_locations[total_weight <= 0]
        raise ValueError(
            f"Total coverage weight is 0 for {empty.size} location(s), "
            f"e.g. {empty[:5].tolist()}."
        )
    return matrix, unique_locations


def build_weights_matrix(
    weights_df: pd.DataFrame
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse zone-by-pixel weights matrix over the weights' unique pixels.

    Pixels shared by several zones are read once, which matters for nested
    basins.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Weights with ``location_id``, ``row``, ``col`` and ``weight`` columns.

    Returns
    -------
    Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]
        The matrix, the ``row`` and ``col`` of each matrix column, and the
        ``location_id`` of each matrix row (sorted).
    """
    rows = weights_df["row"].to_numpy("int64")
    cols = weights_df["col"].to_numpy("int64")
    width = int(cols.max()) + 1
    pixels, pixel_codes = np.unique(rows * width + cols, return_inverse=True)
    matrix, location_ids = _zone_weights_matrix(
        weights_df[LOCATION_ID].to_numpy(),
        pixel_codes,
        pixels.size,
        weights_df["weight"].to_numpy(),
    )
    return matrix, pixels // width, pixels % width, location_ids


def weighted_average_from_matrix(
    matrix: sparse.csr_matrix,
    values: np.ndarray,
    min_coverage: float = 1.0,
) -> np.ndarray:
    """Coverage-weighted mean per zone, ignoring NaN pixels.

    Parameters
    ----------
    matrix : sparse.csr_matrix
        Zone-by-pixel weights, e.g. from :func:`build_weights_matrix`.
    values : np.ndarray
        Pixel values in matrix column order, ``(n_pixels,)`` or
        ``(n_times, n_pixels)``.
    min_coverage : float
        Minimum fraction of a zone's weight that must be on non-NaN pixels;
        below it the zone's value is NaN. The mean is taken over the non-NaN
        pixels. The default 1.0 makes any NaN pixel give NaN.

    Returns
    -------
    np.ndarray
        Means shaped ``(n_zones,)`` or ``(n_times, n_zones)``, matching
        ``values``.
    """
    single_step = np.ndim(values) == 1
    sums, missing_weight = weighted_sums_from_matrix(matrix, values)
    total_weight = matrix @ np.ones(matrix.shape[1])
    means = weighted_means_from_sums(
        sums, missing_weight, total_weight, min_coverage
    )
    return means[0] if single_step else means


def weighted_sums_from_matrix(
    matrix: sparse.spmatrix,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-zone weighted sums of non-NaN pixels, and the weight on NaN pixels.

    Sums from several pixel subsets (e.g. bands of a grid read one at a time)
    can be added before :func:`weighted_means_from_sums`.

    Parameters
    ----------
    matrix : sparse.spmatrix
        Zone-by-pixel weights for the pixels in ``values``.
    values : np.ndarray
        Pixel values in matrix column order, ``(n_pixels,)`` or
        ``(n_times, n_pixels)``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Weighted sums and NaN weight, each ``(n_zones, n_times)``.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim == 1:
        values = values[np.newaxis, :]
    missing = np.isnan(values)
    if missing.any():
        sums = matrix @ np.where(missing, 0.0, values).T
        # Exactly 0 with no NaN, so min_coverage=1.0 never rejects a full zone.
        missing_weight = matrix @ missing.T.astype("float64")
    else:
        sums = matrix @ values.T
        missing_weight = np.zeros_like(sums)
    return sums, missing_weight


def weighted_means_from_sums(
    sums: np.ndarray,
    missing_weight: np.ndarray,
    total_weight: np.ndarray,
    min_coverage: float = 1.0,
) -> np.ndarray:
    """Coverage-weighted means from :func:`weighted_sums_from_matrix` output.

    Parameters
    ----------
    sums, missing_weight : np.ndarray
        ``(n_zones, n_times)`` totals over all of each zone's pixels.
    total_weight : np.ndarray
        Each zone's total weight, ``(n_zones,)``.
    min_coverage : float
        See :func:`weighted_average_from_matrix`.

    Returns
    -------
    np.ndarray
        Means shaped ``(n_times, n_zones)``.
    """
    total_weight = np.asarray(total_weight, dtype="float64")[:, np.newaxis]
    valid_weight = total_weight - missing_weight
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / valid_weight
    means[
        (missing_weight > (1.0 - min_coverage) * total_weight)
        | (valid_weight <= 0)
    ] = np.nan
    return means.T

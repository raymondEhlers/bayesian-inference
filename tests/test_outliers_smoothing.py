"""Tests for standalone smoothing functions.

"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

from bayesian_inference import preprocess_input_data

logger = logging.getLogger(__name__)

_data_dir = Path(__file__).parent / "test_data"


def test_smoothing() -> None:
    # Setup: Load data
    measured_data = np.loadtxt(_data_dir / "tables" / "Data" / "Data__5020__PbPb__hadron__pt_ch_cms____0-5.dat", ndmin=2)
    # Calculate bin centers from data
    x_min = measured_data[:, 0]
    x_max = measured_data[:, 1]
    bin_centers = x_min + (x_max - x_min) / 2.
    # And load values and errors
    values = np.loadtxt(_data_dir / "tables" / "Prediction" / "Prediction__exponential__5020__PbPb__hadron__pt_ch_cms____0-5__values.dat", ndmin=2)
    y_err = np.loadtxt(_data_dir / "tables" / "Prediction" / "Prediction__exponential__5020__PbPb__hadron__pt_ch_cms____0-5__errors.dat", ndmin=2)

    # Identify outliers and smooth them
    output_values, output_y_err, outliers_that_cannot_be_removed = preprocess_input_data.find_and_smooth_outliers_standalone(
        observable_key="hadron__pt_ch_cms",
        bin_centers=bin_centers,
        values=values,
        y_err=y_err,
        # Default values as of September 2024
        outliers_identification_methods={
            "large_statistical_errors": preprocess_input_data.OutliersConfig(n_RMS=2),
            "large_central_value_difference": preprocess_input_data.OutliersConfig(n_RMS=2),
        },
        smoothing_interpolation_method="linear",
        max_n_points_to_interpolate=2,
    )

    assert not np.allclose(output_values, values)
    assert not np.allclose(output_y_err, y_err)

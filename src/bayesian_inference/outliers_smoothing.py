""" Functionality for identifying outliers and smoothing them.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""
from __future__ import annotations

import logging

import attrs
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import]

logger = logging.getLogger(__name__)


IMPLEMENTED_INTERPOLATION_METHODS = ["linear", "cubic_spline"]

@attrs.frozen
class OutliersConfig:
    """Configuration for identifying outliers.

    :param float n_RMS: Number of RMS away from the value to identify as an outlier. Default: 2.
    """
    n_RMS: float = 2.


def find_large_statistical_uncertainty_points(
    values: npt.NDArray[np.float64],
    y_err: npt.NDArray[np.float64],
    outliers_config: OutliersConfig,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Find problematic points based on large statistical uncertainty points.

    Best to do this observable-by-observable because the relative uncertainty will vary for each one.

    Args:
        values: The values of the observable, for all design points.
        y_err: The uncertainties on the values of the observable, for all design points.
        outliers_config: Configuration for identifying outliers.

    Returns:
        (n_feature_index, n_design_point_index) of identified outliers
    """
    relative_error = y_err / values
    # This is the rms averaged over all of the design points
    rms = np.sqrt(np.mean(relative_error**2, axis=-1))
    # NOTE: Recall that np.where returns (n_feature_index, n_design_point_index) as separate arrays
    outliers = np.where(relative_error > outliers_config.n_RMS * rms[:, np.newaxis])
    return outliers  # type: ignore[return-value] # noqa: RET504


def find_outliers_based_on_central_values(
    values: npt.NDArray[np.float64],
    outliers_config: OutliersConfig,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Find outlier points based on large deviations from close central values."""
    # NOTE: We need abs because we don't care about the sign - we just want a measure.
    diff_between_features = np.abs(np.diff(values, axis=0))
    rms = np.sqrt(np.mean(diff_between_features**2, axis=-1))
    outliers_in_diff_mask = (
        diff_between_features > (outliers_config.n_RMS * rms[:, np.newaxis])
    )
    """
    Now, we need to associate the outliers with the original feature index (ie. taking the diff reduces by one)

    The scheme we'll use to identify problematic points is to take an AND of the left and right of the point.
    For the first and last index, we cannot take an and since they're one sided. To address this point, we'll
    redo the exercise, but with the 1th and -2th removed, and take an AND of those and the original. It's ad-hoc,
    but it gives a second level of cross check for those points.
    """
    # First, we'll handle the inner points
    output = np.zeros_like(values, dtype=np.bool_)
    output[1:-1, :] = outliers_in_diff_mask[:-1, :] & outliers_in_diff_mask[1:, :]

    # Convenient breakpoint for debugging of high values
    #if np.any(values > 1.05):
    #    logger.info(f"{values=}")

    # Now, handle the edges. Here, we need to select the 1th and -2th points
    if values.shape[0] > 4:
        s = np.ones(values.shape[0], dtype=np.bool_)
        s[1] = False
        s[-2] = False
        # Now, we'll repeat the calculation with the diff and rMS
        diff_between_features_for_edges = np.abs(np.diff(values[s, :], axis=0))
        rms = np.sqrt(np.mean(diff_between_features_for_edges**2, axis=-1))
        outliers_in_diff_mask_edges = (
            diff_between_features_for_edges > (outliers_config.n_RMS * rms[:, np.newaxis])
        )
        output[0, :] = outliers_in_diff_mask_edges[0, :] & outliers_in_diff_mask[0, :]
        output[-1, :] = outliers_in_diff_mask_edges[-1, :] & outliers_in_diff_mask[-1, :]
    else:
        # Too short - just have to take what we have
        output[0, :] = outliers_in_diff_mask[0, :]
        output[-1, :] = outliers_in_diff_mask[-1, :]

    # NOTE: Recall that np.where returns (n_feature_index, n_design_point_index) as separate arrays
    outliers = np.where(output)
    return outliers  # type: ignore[return-value] # noqa: RET504


def perform_QA_and_reformat_outliers(
    observable_key: str,
    outliers: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
    smoothing_max_n_feature_outliers_to_interpolate: int,
) -> tuple[dict[int, list[int]], dict[str, dict[int, set[int]]]]:
    """ Perform QA on identifier outliers, and reformat them for next steps.

    :param observable_key: The key for the observable we're looking at.
    :param outliers: The outliers provided by the outlier finder.
    :param smoothing_max_n_feature_outliers_to_interpolate: The maximum number of points to interpolate in a row.
    """
    # NOTE: This could skip the observable key, but it's convenient because we then have the same
    #       format as the overall dict
    outliers_we_are_unable_to_remove: dict[str, dict[int, set[int]]] = {}
    # Next, we want to do quality checks.
    # If there are multiple problematic points in a row, we want to skip interpolation since
    # it's not clear that we can reliably interpolate.
    # First, we need to put the features into a more useful order:
    # outliers: zip(feature_index, design_point) -> dict: (design_point, feature_index)
    # NOTE: The `design_point` here is the index in the design point array of the design points
    #       that we've using for this analysis. To actually use them (ie. in print outs), we'll
    #       need to apply them to the actual design point array.
    outlier_features_per_design_point: dict[int, set[int]] = {v: set() for v in outliers[1]}
    for i_feature, design_point in zip(*outliers):
        outlier_features_per_design_point[design_point].update([i_feature])
    # These features must be sorted to finding distances between them, but sets are unordered,
    # so we need to explicitly sort them
    for design_point in outlier_features_per_design_point:
        outlier_features_per_design_point[design_point] = sorted(outlier_features_per_design_point[design_point])  # type: ignore[assignment]

    # Since the feature values of one design point shouldn't impact another, we'll want to
    # check one design point at a time.
    # NOTE: If we have to skip, we record the design point so we can consider excluding it due
    #       to that observable.
    outlier_features_to_interpolate_per_design_point: dict[int, list[int]] = {}
    #logger.info(f"{observable_key=}, {outlier_features_per_design_point=}")
    for k, v in outlier_features_per_design_point.items():
        #logger.debug("------------------------")
        #logger.debug(f"{k=}, {v=}")
        # Calculate the distance between the outlier indices
        distance_between_outliers = np.diff(list(v))
        # And we'll keep track of which ones pass our quality requirements (not too many in a row).
        indices_of_outliers_that_are_one_apart = set()
        accumulated_indices_to_remove = set()

        for distance, lower_feature_index, upper_feature_index in zip(distance_between_outliers, list(v)[:-1], list(v)[1:]):
            # We're only worried about points which are right next to each other
            if distance == 1:
                indices_of_outliers_that_are_one_apart.update([lower_feature_index, upper_feature_index])
            else:
                # In this case, we now have points that aren't right next to each other.
                # Here, we need to figure out what we're going to do with the points that we've found
                # that **are** right next to each other. Namely, we'll want to remove them from the list
                # to be interpolated, but if there are more points than our threshold.
                # NOTE: We want strictly greater than because we add two points per distance being greater than 1.
                #       eg. one distance(s) of 1 -> two points
                #           two distance(s) of 1 -> three points (due to set)
                #           three distance(s) of 1 -> four points (due to set)
                if len(indices_of_outliers_that_are_one_apart) > smoothing_max_n_feature_outliers_to_interpolate:
                    # Since we are looking at the distances, we want to remove the points that make up that distance.
                    accumulated_indices_to_remove.update(indices_of_outliers_that_are_one_apart)
                else:
                    # For debugging, keep track of when we find points that are right next to each other but
                    # where we skip removing them (ie. keep them for interpolation) because they're below our
                    # max threshold of consecutive points
                    # NOTE: There's no point in warning if empty, since that case is trivial
                    if len(indices_of_outliers_that_are_one_apart) > 0:
                        msg = (
                            f"Will continue with interpolating consecutive indices {indices_of_outliers_that_are_one_apart}"
                            f" because the their number is within the allowable range (n_consecutive<={smoothing_max_n_feature_outliers_to_interpolate})."
                        )
                        logger.info(msg)
                # Reset for the next point
                indices_of_outliers_that_are_one_apart = set()
        # There are indices left over at the end of the loop which we need to take care of.
        # eg. If all points are considered outliers
        if indices_of_outliers_that_are_one_apart and \
            len(indices_of_outliers_that_are_one_apart) > smoothing_max_n_feature_outliers_to_interpolate:
            # Since we are looking at the distances, we want to remove the points that make up that distance.
            #logger.info(f"Ended on {indices_of_outliers_that_are_one_apart=}")
            accumulated_indices_to_remove.update(indices_of_outliers_that_are_one_apart)

        # Now that we've determine which points we want to remove from our interpolation (accumulated_indices_to_remove),
        # let's actually remove them from our list.
        # NOTE: We sort again because sets are not ordered.
        outlier_features_to_interpolate_per_design_point[k] = sorted(set(v) - accumulated_indices_to_remove)
        #logger.debug(f"design point {k}: features kept for interpolation: {outlier_features_to_interpolate_per_design_point[k]}")

        # And we'll keep track of what we can't interpolate
        if accumulated_indices_to_remove:
            if observable_key not in outliers_we_are_unable_to_remove:
                outliers_we_are_unable_to_remove[observable_key] = {}
            outliers_we_are_unable_to_remove[observable_key][k] = accumulated_indices_to_remove

    return outlier_features_to_interpolate_per_design_point, outliers_we_are_unable_to_remove


def find_and_smooth_outliers_standalone(
    observable_key: str,
    bin_centers: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    y_err: npt.NDArray[np.float64],
    outliers_identification_methods: dict[str, OutliersConfig],
    smoothing_interpolation_method: str,
    max_n_points_to_interpolate: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[int, set[int]]]:
    """ A standalone function to identify outliers and smooth them.

    Careful: If you remove design points, you'll need to make sure to keep careful track of the indices!

    Note:
        For the outliers that we are unable to remove, it's probably best to exclude the design point entirely.
        However, you'll have to take care of it separately.

    Args:
        observable_key: The key for the observable we're looking at. Just a name for bookkeeping.
        bin_centers: The bin centers for the observable.
        values: The values of the observable, for all design points.
        y_err: The uncertainties on the values of the observable, for all design points.
        outliers_identification_methods: The methods to use for identifying outliers. Keys are the methods, while the values
            are the parameters. Key options: {"large_statistical_errors": OutliersConfig, "large_central_value_difference": OutliersConfig}.
        smoothing_interpolation_method: The method to use for interpolation. Options: ["linear", "cubic_spline"].
        max_n_points_to_interpolate: The maximum number of points to interpolate in a row.

    Returns:
        The smoothed values and uncertainties, and the outliers which we are unable to remove ({feature_index: set(design_point_index)}).
    """
    # Validation
    for outlier_identification_method in outliers_identification_methods:
        if outlier_identification_method not in ["large_statistical_errors", "large_central_value_difference"]:
            msg = f"Unrecognized smoothing method {outlier_identification_method}."
            raise ValueError(msg)
    if len(bin_centers) == 1:
        # Skip - we can't interpolate one point.
        msg = f"Skipping observable \"{observable_key}\" because it has only one point."
        logger.debug(msg)
        raise ValueError(msg)

    # Setup
    outliers_we_are_unable_to_remove: dict[int, set[int]] = {}
    values = np.array(values, copy=True)
    y_err = np.array(y_err, copy=True)

    # Identify outliers
    #outliers = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    outliers = np.zeros((0, 2), dtype=np.int64)
    for outlier_identification_method, outliers_config in outliers_identification_methods.items():
        # First, find the outliers based on the selected method
        if outlier_identification_method == "large_statistical_errors":
            # large statistical uncertainty points
            new_outliers = find_large_statistical_uncertainty_points(
                values=values,
                y_err=y_err,
                outliers_config=outliers_config,
            )
        elif outlier_identification_method == "large_central_value_difference":
            # Find additional outliers based on central values which are dramatically different than the others
            if len(values) > 2:
                new_outliers = find_outliers_based_on_central_values(
                    values=values,
                    outliers_config=outliers_config,
                )
            else:
                new_outliers = ((), ())  # type: ignore[assignment]
        else:
            msg = f"Unrecognized outlier identification mode {outlier_identification_method}."
            raise ValueError(msg)
        # Merge the outliers together, taking care to deduplicate outlier values that may be stored in each array
        combined_indices = np.concatenate((outliers, np.column_stack(new_outliers)), axis=0)
        outliers = np.unique(combined_indices, axis=0)

    # If needed, can split outliers back into the two arrays
    #outliers_feature_indices, outliers_design_point_indices = outliers[:, 0], outliers[:, 0]
    outlier_features_to_interpolate_per_design_point, _intermediate_outliers_we_are_unable_to_remove = perform_QA_and_reformat_outliers(
        observable_key=observable_key,
        outliers=(outliers[:, 0], outliers[:, 1]),
        smoothing_max_n_feature_outliers_to_interpolate=max_n_points_to_interpolate,
    )
    # And keep track of them
    outliers_we_are_unable_to_remove.update(_intermediate_outliers_we_are_unable_to_remove.get(observable_key, {}))

    # Perform interpolation
    for v in [values, y_err]:
        #logger.info(f"Method: {outlier_identification_method}, Interpolating outliers with {outlier_features_to_interpolate_per_design_point=}, {key_type=}, {observable_key=}, {prediction_key=}")
        for design_point, points_to_interpolate in outlier_features_to_interpolate_per_design_point.items():
            try:
                interpolated_values = perform_interpolation_on_values(
                    bin_centers=bin_centers,
                    values_to_interpolate=v[:, design_point],
                    points_to_interpolate=points_to_interpolate,
                    smoothing_interpolation_method=smoothing_interpolation_method,
                )
                # And assign the interpolated values
                v[points_to_interpolate, design_point] = interpolated_values
            except CannotInterpolateDueToOnePointError as e:
                msg = f"Skipping observable \"{observable_key}\", {design_point=} because {e}"
                logger.info(msg)
                # And add to the list since we can't make it work.
                if design_point not in outliers_we_are_unable_to_remove:
                    outliers_we_are_unable_to_remove[design_point] = set()
                outliers_we_are_unable_to_remove[design_point].update(points_to_interpolate)
                continue

    return values, y_err, outliers_we_are_unable_to_remove



class CannotInterpolateDueToOnePointError(Exception):
    """ Error raised when we can't interpolate due to only one point. """


def perform_interpolation_on_values(
    bin_centers: npt.NDArray[np.float64],
    values_to_interpolate: npt.NDArray[np.float64],
    points_to_interpolate: list[int],
    smoothing_interpolation_method: str,
) -> npt.NDArray[np.float64]:
    """ Perform interpolation on the requested points to interpolate.

    Args:
        bin_centers: The bin centers for the observable.
        values_to_interpolate: The values to interpolate.
        points_to_interpolate: The points (i.e. bin centers) to interpolate.
        smoothing_interpolation_method: The method to use for interpolation. Options:
            ["linear", "cubic_spline"].

    Returns:
        The values that are interpolated at points_to_interpolate. They cna be inserted into the
            original values_to_interpolate array via `values_to_interpolate[points_to_interpolate] = interpolated_values`.

    Raises:
        CannotInterpolateDueToOnePointError: Raised when we can't interpolate due to only
            one point being left.
    """
    # Validation for methods
    if smoothing_interpolation_method not in IMPLEMENTED_INTERPOLATION_METHODS:
        msg = f"Unrecognized interpolation method {smoothing_interpolation_method}."
        raise ValueError(msg)

    # We want to train the interpolation only on all good points, so we take them out.
    # Otherwise, it will negatively impact the interpolation.
    mask = np.ones_like(bin_centers, dtype=bool)
    mask[points_to_interpolate] = False

    # Further validation
    if len(bin_centers[mask]) == 1:
        # Skip - we can't interpolate one point.
        msg = f"Can't interpolate due to only one point left to anchor the interpolation. {mask=}"
        raise CannotInterpolateDueToOnePointError(msg)

    # NOTE: ROOT::Interpolator uses a Cubic Spline, so this might be a reasonable future approach
    #       However, I think it's slower, so we'll start with this simple approach.
    # TODO: We entirely ignore the interpolation error here. Some approaches for trying to account for it:
    #       - Attempt to combine the interpolation error with the statistical error
    #       - Randomly remove a few percent of the points which are used for estimating the interpolation,
    #         and then see if there are significant changes in the interpolated parameters
    #       - Could vary some parameters (perhaps following the above) and perform the whole
    #         Bayesian analysis, again looking for how much the determined parameters change.
    if smoothing_interpolation_method == "linear":
        interpolated_values = np.interp(
            bin_centers[points_to_interpolate],
            bin_centers[mask],
            values_to_interpolate[mask],
        )
    elif smoothing_interpolation_method == "cubic_spline":
        cs = scipy.interpolate.CubicSpline(
            bin_centers[mask],
            values_to_interpolate[mask],
        )
        interpolated_values = cs(bin_centers[points_to_interpolate])

    return interpolated_values


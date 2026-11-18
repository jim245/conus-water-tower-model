"""
Computes RWY (Runoff Water Yield) and WRC (Water Resource Contribution) for each hydrologic unit of a basin, at the
huc4, huc8 and huc12 levels.

main.py compares each upstream cell of a basin with one downstream outlet region. This module compares each cell with
the descendants D_A of its unit A: all the units downstream of A in the flow graph (wrd_flows.build_flow_graph,
wrd_flows.sum_descendants). WRD (wrd_flows.py) uses the ancestors of a unit. RWY and WRC use its descendants:

    RWY_cell = q_cell / mean(q over D_A)
    WRC_cell = (mean(Q over D_A) - mean(WW over D_A)) / (qi_cell - wc_cell)

    q, Q, qi, WW, wc = the inputs of main.run_rwy and main.run_wrc, clipped to the basin

A mean over D_A is the mean of all the cells of all the descendants, not a mean of the unit means. The cells are the
cells of utilities.safe_mean. A unit without descendants (an outlet or a closed-basin sink) gets no RWY and no WRC.
The flow graph gives the descendants, with no exclusion by hand.

main.main calls run_year for each basin, scenario and year. conus_maps.run_network_rwy_wrc makes the CONUS maps.
"""

from pathlib import Path
from typing import Any

import geopandas
import numpy

import index_maps
import raster_preparation
import utilities
import wrd_flows


def unit_sums(labels: numpy.ndarray, data: numpy.ndarray, n_labels: int) -> numpy.ndarray:
    """
    Sums the valid cells of each label, and counts them. A valid cell is a cell of utilities.safe_mean.

    Parameters:
        labels (numpy.ndarray): The unit labels (wrd_flows.rasterize_units).
        data (numpy.ndarray): The raster data, on the grid of labels.
        n_labels (int): The number of labels, with label 0.

    Returns:
        sums (numpy.ndarray): An (n_labels, 2) array: the sum and the count of each label.
    """

    valid = numpy.isfinite(data) & (data >= 0)
    flat_labels = labels.ravel()
    return numpy.stack(
        [
            numpy.bincount(flat_labels, weights=numpy.where(valid, data, 0.0).ravel(), minlength=n_labels),
            numpy.bincount(flat_labels, weights=valid.ravel().astype(float), minlength=n_labels),
        ],
        axis=1,
    )


def descendant_means(
    graph: dict[str, list[str]],
    unit_codes: list[str],
    labels: numpy.ndarray,
    rasters: list[numpy.ndarray],
) -> numpy.ndarray:
    """
    Computes the mean of each raster over the descendants of each unit.

    Parameters:
        graph (dict[str, list[str]]): The flow graph (wrd_flows.build_flow_graph).
        unit_codes (list[str]): The unit code of each label, from label 1.
        labels (numpy.ndarray): The unit labels (wrd_flows.rasterize_units).
        rasters (list[numpy.ndarray]): The raster data, on the grid of labels.

    Returns:
        means (numpy.ndarray): An (n_labels, len(rasters)) array. NaN for label 0 (no unit), for a unit without
            descendants and where the descendants have no valid cell.
    """

    n_labels = len(unit_codes) + 1
    sums = numpy.concatenate([unit_sums(labels, r, n_labels) for r in rasters], axis=1)
    totals = wrd_flows.sum_descendants(graph, {code: sums[label] for label, code in enumerate(unit_codes, start=1)})

    means = numpy.full((n_labels, len(rasters)), numpy.nan)
    for label, code in enumerate(unit_codes, start=1):
        if not graph.get(code):
            continue
        total = totals[code]
        with numpy.errstate(divide="ignore", invalid="ignore"):
            means[label] = total[0::2] / total[1::2]
    means[~numpy.isfinite(means)] = numpy.nan
    return means


def compute_rwy(
    quickflow: numpy.ndarray,
    labels: numpy.ndarray,
    unit_codes: list[str],
    graph: dict[str, list[str]],
) -> numpy.ndarray:
    """
    Computes the RWY of each cell against the descendants of its unit (see the module docstring).

    Parameters:
        quickflow (numpy.ndarray): The mean Quickflow, float64, NaN for no data.
        labels (numpy.ndarray): The unit labels on the Quickflow grid (wrd_flows.rasterize_units).
        unit_codes (list[str]): The unit code of each label, from label 1.
        graph (dict[str, list[str]]): The flow graph (wrd_flows.build_flow_graph).

    Returns:
        rwy (numpy.ndarray): The RWY of each cell, float64. NaN where the mean over the descendants is not positive.
    """

    downstream_average = descendant_means(graph, unit_codes, labels, [quickflow])[:, 0][labels]

    rwy = numpy.full(quickflow.shape, numpy.nan, dtype="float64")
    valid = numpy.isfinite(quickflow) & (downstream_average > 0)
    rwy[valid] = quickflow[valid] / downstream_average[valid]
    return rwy


def compute_wrc(
    runoff: numpy.ndarray,
    consumption: numpy.ndarray,
    withdrawal: numpy.ndarray,
    labels: numpy.ndarray,
    unit_codes: list[str],
    graph: dict[str, list[str]],
) -> numpy.ndarray:
    """
    Computes the WRC of each cell against the descendants of its unit (see the module docstring).

    Inside the runoff area, a water-use cell with no value counts as 0
    (raster_preparation.prepare_pws_withdrawal_raster).

    Parameters:
        runoff (numpy.ndarray): The runoff, in mm/yr, NaN for no data.
        consumption (numpy.ndarray): The consumption, in mm/yr, NaN for no data, on the same grid.
        withdrawal (numpy.ndarray): The withdrawal, in mm/yr, NaN for no data, on the same grid.
        labels (numpy.ndarray): The unit labels on the same grid (wrd_flows.rasterize_units).
        unit_codes (list[str]): The unit code of each label, from label 1.
        graph (dict[str, list[str]]): The flow graph (wrd_flows.build_flow_graph).

    Returns:
        wrc (numpy.ndarray): The WRC of each cell. NaN where the runoff or the mean over the descendants has no value,
            and where |qi - wc| is less than utilities.WRC_MIN_DENOMINATOR.
    """

    withdrawal = numpy.where(numpy.isfinite(runoff) & numpy.isnan(withdrawal), 0.0, withdrawal)
    consumption = numpy.where(numpy.isfinite(runoff) & numpy.isnan(consumption), 0.0, consumption)

    means = descendant_means(graph, unit_codes, labels, [runoff, withdrawal])
    numerator = (means[:, 0] - means[:, 1])[labels]

    denominator = runoff - consumption
    denominator[numpy.abs(denominator) < utilities.WRC_MIN_DENOMINATOR] = numpy.nan

    with numpy.errstate(divide="ignore", invalid="ignore"):
        wrc = numerator / denominator
    wrc[~numpy.isfinite(runoff) | ~numpy.isfinite(consumption)] = numpy.nan
    wrc[~numpy.isfinite(wrc)] = numpy.nan
    return wrc


def run_year(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: Path,
    networks: list[dict[str, Any]],
    rasters: wrd_flows.BasinRasters,
) -> None:
    """
    Computes and writes the RWY and the WRC of one scenario and year at each level. Prints the statistics of each
    level.

    Writes quickflow/<scenario>/<scenario>_<year>_averaged_basin.tif, and <scenario>_<year>_rwy.tif and .png in
    rwy/huc<level>/<scenario>/ and <scenario>_<year>_wrc.tif and .png in wrc/huc<level>/<scenario>/, under
    root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons of the basin, for the clip and the map outlines.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year.
        root_output_directory (Path): The output directory of the basin.
        networks (list[dict[str, Any]]): The networks (wrd_flows.build_level).
        rasters (wrd_flows.BasinRasters): The rasters of the scenario and year (wrd_flows.prepare_basin_rasters).
    """

    quickflow_directory = utilities.create_directory(Path(root_output_directory, "quickflow", climate_model))
    averaged, averaged_profile, _ = raster_preparation.read_national_raster(
        raster_preparation.national_quickflow_path(climate_model, year)
    )
    quickflow, quickflow_profile, _ = utilities.clip_raster(
        averaged,
        averaged_profile,
        hucs.geometry,
        Path(quickflow_directory, f"{climate_model}_{year}_averaged_basin.tif"),
    )
    runoff, consumption, withdrawal, profile = rasters

    for network in networks:
        level = network["level"]
        rwy_directory = utilities.create_directory(Path(root_output_directory, "rwy", f"huc{level}", climate_model))
        wrc_directory = utilities.create_directory(Path(root_output_directory, "wrc", f"huc{level}", climate_model))

        rwy = compute_rwy(
            quickflow.astype("float64"),
            wrd_flows.unit_labels(network, quickflow_profile),
            network["unit_codes"],
            network["graph"],
        )
        rwy, rwy_profile, _ = utilities.save_raster(
            rwy,
            quickflow_profile,
            Path(rwy_directory, f"{climate_model}_{year}_rwy.tif"),
        )
        utilities.print_stats(rwy, f"RWY huc{level} ({climate_model}, {year})")
        index_maps.plot_rwy_classes(
            rwy,
            rwy_profile,
            hucs,
            Path(rwy_directory, f"{climate_model}_{year}_rwy.png"),
            title=f"RWY huc{level} ({climate_model}): {year}",
        )

        wrc = compute_wrc(
            runoff,
            consumption,
            withdrawal,
            wrd_flows.unit_labels(network, profile),
            network["unit_codes"],
            network["graph"],
        )
        wrc, wrc_profile, _ = utilities.save_raster(
            wrc,
            profile,
            Path(wrc_directory, f"{climate_model}_{year}_wrc.tif"),
        )
        utilities.print_stats(wrc, f"WRC huc{level} ({climate_model}, {year})")
        index_maps.plot_wrc_classes(
            wrc,
            wrc_profile,
            hucs,
            Path(wrc_directory, f"{climate_model}_{year}_wrc.png"),
            title=f"WRC huc{level} ({climate_model}): {year}",
        )

"""
Computes RWY (Relative Water Yield), WRC (Water Resource Contribution) and WRD (Water Resource Dependency) for each
basin of CONUS_BASINS. The scenarios are CLIMATE_MODELS and the years are YEARS.

RWY and WRC compare each upstream cell of a basin with one downstream outlet region D (BASIN_MAPPINGS):

    RWY_cell = q_cell / mean(q over D)
    WRC_cell = (mean(Q over D) - mean(WW over D)) / (qi_cell - wc_cell)

    q      = the mean Quickflow over the GCMs, on the Quickflow grid
    Q, qi  = the same Quickflow in mm/yr, on the 1 km water-use grid
    WW     = the surface-water public-supply withdrawal (pws-sw), in mm/yr
    wc     = the consumption (irrigation, pws-cu and thermoelectric), in mm/yr

The means are over the finite, non-negative cells (utilities.safe_mean). WW has no 0.75 factor: the original method
uses it for the Potomac study only.

For each basin, WRD (wrd_flows.run_year) and RWY and WRC at each network level (network_indices.run_year) follow.
Writes output/<basin>/. Run conus_maps.py after this module for the CONUS maps.
"""

import os

# conda activation sets PROJ_NETWORK=ON. PROJ can then download grid shifts, and a transformation then depends on the
# network access and on PROJ's grid cache. Set it before the imports that load PROJ
os.environ["PROJ_NETWORK"] = "OFF"

# isort: split

from pathlib import Path

import geopandas
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy

import index_maps
import network_indices
import raster_preparation
import utilities
import wrd_flows

# basin: (huc2s, huc4s excluded from the upstream area, huc4s of the downstream outlet region). A huc4 that is not in
# the flow network of the basin goes in the excluded list and not in the outlet region (0807 and 0808 in the MRB, 1508
# in the CRB)
BASIN_MAPPINGS = {
    "NER": (
        ["01"],
        ["0101", "0105", "0106", "0108", "0109", "0110"],
        ["0101", "0105", "0106", "0108", "0109", "0110"],
    ),  # New England Region
    "MAR": (
        ["02"],
        ["0203", "0204", "0208"],
        ["0203", "0204", "0208"],
    ),  # Mid-Atlantic Region
    "SAG": (
        ["03"],
        [
            "0301",
            "0302",
            "0303",
            "0304",
            "0305",
            "0306",
            "0307",
            "0308",
            "0310",
            "0311",
            "0312",
            "0313",
            "0314",
            "0317",
            "0318",
        ],
        [
            "0301",
            "0302",
            "0303",
            "0304",
            "0305",
            "0306",
            "0307",
            "0308",
            "0310",
            "0311",
            "0312",
            "0313",
            "0314",
            "0317",
            "0318",
        ],
    ),  # South Atlantic-Gulf Region
    "GLR": (["04"], ["0429", "0430", "0431"], ["0429", "0430", "0431"]),  # Great Lakes Region
    "OHR": (["05"], ["0514"], ["0514"]),  # Ohio Region
    "TNR": (["06"], ["0604"], ["0604"]),  # Tennessee Region
    "UMR": (["07"], ["0714"], ["0714"]),  # Upper Mississippi Region
    "LMR": (["08"], ["0809"], ["0809"]),  # Lower Mississippi Region
    "MOR": (["10"], ["1030"], ["1030"]),  # Missouri Region
    "AWR": (
        ["11"],
        ["1101", "1111", "1114"],
        ["1101", "1111", "1114"],
    ),  # Arkansas-White-Red Region
    "MRB": (
        ["05", "06", "07", "08", "10", "11"],
        ["0807", "0808", "0809"],
        ["0809"],
    ),  # Mississippi River Basin
    "TGR": (
        ["12"],
        ["1204", "1209", "1210", "1211"],
        ["1204", "1209", "1210", "1211"],
    ),  # Texas-Gulf Region
    "RGR": (["13"], ["1305", "1309"], ["1305", "1309"]),  # Rio Grande Region
    "UCR": (["14"], ["1408"], ["1408"]),  # Upper Colorado Region
    "LCR": (["15"], ["1503", "1508"], ["1503"]),  # Lower Colorado Region
    "CRB": (["14", "15"], ["1503", "1508"], ["1503"]),  # Colorado River Basin
    "GBR": (
        ["16"],
        ["1602", "1603", "1605", "1606"],
        ["1602", "1603", "1605", "1606"],
    ),  # Great Basin Region
    "PNW": (
        ["17"],
        ["1708", "1710", "1711", "1712"],
        ["1708", "1710", "1711", "1712"],
    ),  # Pacific Northwest Region
    "CAR": (
        ["18"],
        ["1801", "1803", "1805", "1806", "1807", "1808", "1809", "1810"],
        ["1801", "1803", "1805", "1806", "1807", "1808", "1809", "1810"],
    ),  # California Region
}

# the basins of the CONUS mosaic, with no overlap. The Mississippi and the Colorado are whole river systems (MRB, CRB),
# not their huc2 regions, so that each cell has its real outlet. huc2 09 (Souris-Red-Rainy) is in no basin and is a gap
# in the mosaic
CONUS_BASINS = [
    "NER",
    "MAR",
    "SAG",
    "GLR",
    "MRB",
    "TGR",
    "RGR",
    "CRB",
    "GBR",
    "PNW",
    "CAR",
]

CLIMATE_MODELS = ["SSP245", "SSP370"]
YEARS = list(range(2015, 2020 + 1))  # the diff maps (first and last year) need 2 or more years


def run_rwy(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    basin_hucs: geopandas.GeoDataFrame,
    root_output_directory: Path,
) -> None:
    """
    Computes the RWY of each upstream cell for each scenario and year, and the diff of the first and the last year.

    The RWY has no value in all cells if the mean downstream Quickflow is not positive: then the function prints a
    warning. The color scale of the diff map is from the 2nd to the 98th percentile. Prints the statistics of each
    year.

    Writes rwy/<scenario>/<scenario>_<year>_rwy.tif and .png, and <scenario>_diff_rwy.tif and .png, under
    root_output_directory (and the steps of raster_preparation.prepare_quickflow_rasters).

    Parameters:
        upstream_hucs (geopandas.GeoDataFrame): The huc4 polygons of the upstream area.
        downstream_hucs (geopandas.GeoDataFrame): The huc4 polygons of the downstream outlet region.
        basin_hucs (geopandas.GeoDataFrame): The huc4 polygons of the basin, for the map outlines.
        root_output_directory (Path): The output directory of the basin.
    """

    for climate_model in CLIMATE_MODELS:
        result_arrays = list[numpy.ndarray]()
        output_directory = utilities.create_directory(Path(root_output_directory, "rwy", climate_model))

        profile = None
        for year in YEARS:
            upstream, downstream = raster_preparation.prepare_quickflow_rasters(
                upstream_hucs,
                downstream_hucs,
                climate_model,
                year,
                root_output_directory,
            )
            downstream_data, _, _ = downstream
            upstream_data, profile, _ = upstream

            downstream_average = utilities.safe_mean(downstream_data, numpy.nan)

            rwy = numpy.full(upstream_data.shape, numpy.nan, dtype="float64")
            if downstream_average > 0:
                valid_upstream = numpy.isfinite(upstream_data)
                rwy[valid_upstream] = upstream_data[valid_upstream] / downstream_average
            else:
                print(
                    f"  WARNING: RWY ({climate_model}, {year}): the mean downstream Quickflow is {downstream_average}, "
                    "not positive. The RWY has no value"
                )

            rwy, profile, _ = utilities.save_raster(
                rwy,
                profile,
                Path(output_directory, f"{climate_model}_{year}_rwy.tif"),
            )
            result_arrays.append(rwy)
            utilities.print_stats(rwy, f"RWY ({climate_model}, {year})")

            index_maps.plot_rwy_classes(
                rwy,
                profile,
                basin_hucs,
                Path(output_directory, f"{climate_model}_{year}_rwy.png"),
                title=f"RWY ({climate_model}): {year}",
            )

        assert profile is not None

        if len(YEARS) < 2:
            continue

        diff, profile, _ = utilities.compute_diff(
            result_arrays[0],
            result_arrays[-1],
            profile,
            Path(output_directory, f"{climate_model}_diff_rwy.tif"),
        )

        finite = diff[numpy.isfinite(diff)]
        if finite.size == 0:
            print(f"  WARNING: RWY diff ({climate_model}) has no value, no diff map")
            continue
        vmin, vmax = numpy.nanpercentile(finite, [2, 98])

        utilities.plot_raster_map(
            diff,
            profile,
            basin_hucs,
            Path(output_directory, f"{climate_model}_diff_rwy.png"),
            title=f"RWY ({climate_model}): {YEARS[0]}-{YEARS[-1]}",
            cmap=plt.get_cmap("YlOrRd").copy(),
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
            cbar_label="Relative Water Yield (RWY)",
        )


def run_wrc(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    basin_hucs: geopandas.GeoDataFrame,
    root_output_directory: Path,
) -> None:
    """
    Computes the WRC of each upstream cell for each scenario and year, and the diff of the first and the last year.

    The runoff depends on the scenario. The water use does not. Inside the runoff area, a water-use cell with no value
    counts as 0 (raster_preparation.prepare_pws_withdrawal_raster). The color scale of the diff map is from the 2nd to
    the 98th percentile. Prints the statistics of each year.

    Writes wrc/<scenario>/<scenario>_<year>_wrc.tif and .png, <scenario>_diff_wrc.tif and .png, and
    <year>-consumption_total.tif under root_output_directory (and the steps of the raster_preparation functions).

    Parameters:
        upstream_hucs (geopandas.GeoDataFrame): The huc4 polygons of the upstream area.
        downstream_hucs (geopandas.GeoDataFrame): The huc4 polygons of the downstream outlet region.
        basin_hucs (geopandas.GeoDataFrame): The huc4 polygons of the basin, for the map outlines.
        root_output_directory (Path): The output directory of the basin.
    """

    for climate_model in CLIMATE_MODELS:
        result_arrays = list[numpy.ndarray]()
        output_directory = utilities.create_directory(Path(root_output_directory, "wrc", climate_model))

        profile = None
        for year in YEARS:
            upstream_runoff, downstream_runoff = raster_preparation.prepare_runoff_rasters(
                upstream_hucs,
                downstream_hucs,
                climate_model,
                year,
                root_output_directory,
            )

            irr = raster_preparation.prepare_irr_rasters(
                upstream_hucs, root_output_directory, year, region_tag="upstream"
            )
            upstream_pws, downstream_pws = raster_preparation.prepare_pws_rasters(
                upstream_hucs, downstream_hucs, root_output_directory, year
            )
            thermoelectric = raster_preparation.prepare_thermoelectric_rasters(
                upstream_hucs,
                root_output_directory,
                year,
                region_tag="upstream",
                reference_raster_path=irr[2],
            )

            consumption_sum = utilities.sum_rasters(
                [irr, upstream_pws, thermoelectric],
                Path(root_output_directory, f"{year}-consumption_total.tif"),
            )

            qi, qi_profile = upstream_runoff[:2]
            wc, wc_profile = consumption_sum[:2]
            qd, qd_profile = downstream_runoff[:2]
            ww, ww_profile = downstream_pws[:2]

            ww = numpy.where(numpy.isfinite(qd) & numpy.isnan(ww), 0.0, ww)
            wc = numpy.where(numpy.isfinite(qi) & numpy.isnan(wc), 0.0, wc)

            runoff_mean = utilities.safe_mean(qd, qd_profile["nodata"])
            withdrawal_mean = utilities.safe_mean(ww, ww_profile["nodata"])

            numerator = runoff_mean - withdrawal_mean

            denominator = qi - wc
            denominator[numpy.abs(denominator) < utilities.WRC_MIN_DENOMINATOR] = numpy.nan

            wrc = numerator / denominator

            print(f"  WRC ({climate_model}, {year}):")
            print(f"  Q_downstream_avg:  {runoff_mean:.3f} mm/yr")
            print(f"  WW_downstream_avg: {withdrawal_mean:.3f} mm/yr")
            print(f"  Scalar numerator:  {numerator:.3f} mm/yr")
            print(f"  Denominator range: {numpy.nanmin(denominator):.3f} to {numpy.nanmax(denominator):.3f} mm/yr")
            print(f"  WRC range (pre-mask): {numpy.nanmin(wrc):.3f} to {numpy.nanmax(wrc):.3f}")

            nodata_mask = ~utilities.valid_mask(qi, qi_profile["nodata"]) | ~utilities.valid_mask(
                wc, wc_profile["nodata"]
            )
            wrc[nodata_mask] = numpy.nan
            wrc[~numpy.isfinite(wrc)] = numpy.nan

            wrc, profile, _ = utilities.save_raster(
                wrc,
                qi_profile,
                Path(output_directory, f"{climate_model}_{year}_wrc.tif"),
            )
            result_arrays.append(wrc)
            utilities.print_stats(wrc, f"WRC ({climate_model}, {year})")

            index_maps.plot_wrc_classes(
                wrc,
                profile,
                basin_hucs,
                Path(output_directory, f"{climate_model}_{year}_wrc.png"),
                title=f"WRC ({climate_model}): {year}",
            )

        assert profile is not None

        if len(YEARS) < 2:
            continue

        diff, profile, _ = utilities.compute_diff(
            result_arrays[0],
            result_arrays[-1],
            profile,
            Path(output_directory, f"{climate_model}_diff_wrc.tif"),
        )

        finite = diff[numpy.isfinite(diff)]
        if finite.size == 0:
            print(f"  WARNING: WRC diff ({climate_model}) has no value, no diff map")
            continue
        vmin, vmax = numpy.nanpercentile(finite, [2, 98])

        utilities.plot_raster_map(
            diff,
            profile,
            basin_hucs,
            Path(output_directory, f"{climate_model}_diff_wrc.png"),
            title=f"WRC ({climate_model}): {YEARS[0]}-{YEARS[-1]}",
            cmap=plt.get_cmap("viridis").copy(),
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
            cbar_label="Water Resource Contribution (WRC)",
        )


def main() -> None:
    """
    Computes and writes RWY, WRC, WRD and the network RWY and WRC of each basin of CONUS_BASINS, in output/<basin>/.

    First computes the national rasters (raster_preparation.prepare_national_rasters), one time for all basins. The huc4
    polygons are the GDB huc12s dissolved by code (wrd_flows.load_units): the polygons of the flow network. For each
    basin, builds the networks one time. Prepares the water use one time for each year, and the runoff one time for
    each scenario and year. The WRD and the network RWY and WRC use the same networks and rasters.

    Raises:
        ValueError: If a huc4 of BASIN_MAPPINGS is not a huc4 of the basin in the GDB.
    """

    raster_preparation.prepare_national_rasters(CLIMATE_MODELS, YEARS)

    for basin in CONUS_BASINS:
        huc2s, excluded_upstream_huc4s, downstream_huc4s = BASIN_MAPPINGS[basin]

        huc12s = wrd_flows.load_huc12s(huc2s)
        units = {level: wrd_flows.load_units(huc12s, level) for level in wrd_flows.LEVELS}
        basin_hucs = units[4]

        unknown_huc4s = sorted(set(excluded_upstream_huc4s + downstream_huc4s) - set(basin_hucs["unit"]))
        if unknown_huc4s:
            raise ValueError(
                f"{basin}: the huc4s {unknown_huc4s} of BASIN_MAPPINGS are not huc4s of the basin in the GDB"
            )

        upstream_hucs = basin_hucs[~basin_hucs["unit"].isin(excluded_upstream_huc4s)]
        downstream_hucs = basin_hucs[basin_hucs["unit"].isin(downstream_huc4s)]

        output_directory = utilities.create_directory(Path("output", basin))

        run_rwy(upstream_hucs, downstream_hucs, basin_hucs, output_directory)
        run_wrc(upstream_hucs, downstream_hucs, basin_hucs, output_directory)

        networks = [wrd_flows.build_level(huc12s, units[level], level) for level in wrd_flows.LEVELS]
        for year in YEARS:
            water_use = wrd_flows.prepare_water_use(basin_hucs, year, output_directory)
            for climate_model in CLIMATE_MODELS:
                rasters = wrd_flows.prepare_basin_rasters(basin_hucs, climate_model, year, output_directory, water_use)
                wrd_flows.run_year(basin_hucs, climate_model, year, output_directory, networks, rasters)
                network_indices.run_year(basin_hucs, climate_model, year, output_directory, networks, rasters)


if __name__ == "__main__":
    main()

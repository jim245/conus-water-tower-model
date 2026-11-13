from os import path
from typing import Any

import geopandas
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import pandas

import raster_preparation
import utilities
import wrd_flows

# TODO: switch total water use to surface only
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
    "GLR": (["04"], ["0415"], ["0415"]),  # Great Lakes Region
    "OHR": (["05"], ["0514"], ["0514"]),  # Ohio Region
    "TNR": (["06"], ["0604"], ["0604"]),  # Tennessee Region
    "UMR": (["07"], ["0714"], ["0714"]),  # Upper Mississippi Region
    "LMR": (["08"], ["0809"], ["0809"]),  # Lower Mississippi Region
    # "SRR": (
    #     ["09"],
    #     ["0901", "0902", "0903", "0904"],
    #     ["0901", "0902", "0903", "0904"],
    # ),  # Souris-Red-Rainy Region
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
    # "AKR": (
    #     ["19"],
    #     ["1901", "1902", "1903", "1905", "1906", "1909"],
    #     ["1901", "1902", "1903", "1905", "1906", "1909"],
    # ),  # Alaska Region
    # "HIR": (
    #     ["20"],
    #     ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008"],
    #     ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008"],
    # ),  # Hawaii Region
    # "CRR": (["21"], ["2101", "2102"], ["2101", "2102"]),  # Caribbean Region
    # "SPR": (
    #     ["22"],
    #     ["2201", "2202", "2203"],
    #     ["2201", "2202", "2203"],
    # ),  # South Pacific Region
}

# Non-overlapping set of BASIN_MAPPINGS keys that tiles CONUS: the Mississippi and
# Colorado are run as whole river systems (MRB, CRB) rather than their HUC2 regions,
# so every cell is evaluated against its real outlet. SRR (09) is commented out above,
# so it's a gap in the mosaic.
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
YEARS = list(range(2015, 2020 + 1))  # diffs (first vs last year) need 2+ years


RWY_BOUNDARIES = [0, 0.5, 1, 2, 3, 5, 10]
RWY_COLORS = [
    "lightgray",  # = 0
    "#f5d300",  # 0 to 0.5: yellow
    "#fff3a0",  # 0.5 to 1: lighter yellow
    "#f9b4c8",  # 1 to 2: pink
    "#e7609a",  # 2 to 3: dark pink
    "#e31a1c",  # 3 to 5: red
    "#b10026",  # 5 to 10: darker red
    "#67000d",  # > 10: dark red
]
RWY_LABELS = [
    "0",
    "0 to 0.5",
    "0.5 to 1",
    "1 to 2",
    "2 to 3",
    "3 to 5",
    "5 to 10",
    "> 10",
]


def plot_rwy_classes(
    rwy: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: str,
    title: str,
):
    # Upper-inclusive bins (0 < RWY <= 0.5, ...), plus a class of its own for 0:
    # cells with no quickflow at all, common across the desert Southwest.
    rwy_classes = numpy.full(rwy.shape, numpy.nan, dtype="float32")
    rwy_classes[rwy <= 0] = 0
    for i, (low, high) in enumerate(zip(RWY_BOUNDARIES, RWY_BOUNDARIES[1:])):
        rwy_classes[(rwy > low) & (rwy <= high)] = i + 1
    rwy_classes[rwy > RWY_BOUNDARIES[-1]] = len(RWY_BOUNDARIES)

    counts = {label: int(numpy.sum(rwy_classes == i)) for i, label in enumerate(RWY_LABELS)}
    print(f"  RWY category counts: {counts}")

    n_classes = len(RWY_COLORS)
    utilities.plot_raster_map(
        rwy_classes,
        profile,
        hucs,
        output_path,
        title=title,
        cmap=mcolors.ListedColormap(RWY_COLORS),
        norm=mcolors.BoundaryNorm(boundaries=numpy.arange(n_classes + 1) - 0.5, ncolors=n_classes),
        cbar_label="Runoff Water Yield (RWY)",
        cbar_ticks=list(range(n_classes)),
        cbar_ticklabels=RWY_LABELS,
    )


def run_rwy(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    basin_hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
):
    for climate_model in CLIMATE_MODELS:
        result_arrays = list[numpy.ndarray]()
        output_directory = utilities.create_directory(
            path.join(root_output_directory, "rwy", climate_model), exist_ok=True
        )

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

            rwy, profile, _ = utilities.save_raster(
                rwy,
                profile,
                path.join(output_directory, f"{climate_model}_{year}_rwy.tif"),
            )
            result_arrays.append(rwy)

            plot_rwy_classes(
                rwy,
                profile,
                basin_hucs,
                path.join(output_directory, f"{climate_model}_{year}_rwy.png"),
                title=f"RWY ({climate_model}): {year}",
            )

            # TODO: stats and histogram

        assert profile is not None

        if len(YEARS) < 2:
            continue

        diff, profile, _ = utilities.compute_diff(
            result_arrays[0],
            result_arrays[-1],
            profile,
            path.join(output_directory, f"{climate_model}_diff_rwy.tif"),
        )

        finite = diff[numpy.isfinite(diff)]
        vmin, vmax = numpy.nanpercentile(finite, [2, 98])

        utilities.plot_raster_map(
            diff,
            profile,
            basin_hucs,
            path.join(output_directory, f"{climate_model}_diff_rwy.png"),
            title=f"RWY ({climate_model}): 2015-2020",
            cmap=plt.get_cmap("YlOrRd").copy(),
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
            cbar_label="Runoff Water Yield (RWY)",
        )


def plot_wrc_classes(
    wrc: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: str,
    title: str,
):
    # Classify into the same 4 bins as the original pipeline's plot_wrc:
    # magnitude within a bin doesn't matter, only which side of these
    # thresholds a cell falls on.
    wrc_classes = numpy.full(wrc.shape, numpy.nan, dtype="float32")
    wrc_classes[wrc <= -1] = 0
    wrc_classes[(wrc > -1) & (wrc < 0)] = 1
    wrc_classes[(wrc >= 0) & (wrc <= 1)] = 2
    wrc_classes[wrc > 1] = 3

    print(
        "  WRC category counts: "
        f"{{'≤ -1': {int(numpy.sum(wrc <= -1))}, "
        f"'-1 to 0': {int(numpy.sum((wrc > -1) & (wrc < 0)))}, "
        f"'0 to 1': {int(numpy.sum((wrc >= 0) & (wrc <= 1)))}, "
        f"'> 1': {int(numpy.sum(wrc > 1))}}}"
    )

    wrc_cmap = mcolors.ListedColormap(["orange", "yellow", "green", "lightblue"])
    wrc_norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    utilities.plot_raster_map(
        wrc_classes,
        profile,
        hucs,
        output_path,
        title=title,
        cmap=wrc_cmap,
        norm=wrc_norm,
        cbar_label="Water Resource Contribution (WRC)",
        cbar_ticks=[0, 1, 2, 3],
        cbar_ticklabels=["≤ -1", "-1 to 0", "0 to 1", "> 1"],
    )


def run_wrc(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    basin_hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
):
    for climate_model in CLIMATE_MODELS:
        result_arrays = list[numpy.ndarray]()
        output_directory = utilities.create_directory(
            path.join(root_output_directory, "wrc", climate_model), exist_ok=True
        )

        profile = None
        for year in YEARS:
            # Runoff is the GCM-averaged quickflow for this scenario/year (the same
            # source RWY uses); water use doesn't depend on the climate model.
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
                path.join(root_output_directory, f"{year}-consumption_total.tif"),
            )

            # compute wrc
            qi, qi_profile = upstream_runoff[:2]
            wc, wc_profile = consumption_sum[:2]
            qd, qd_profile = downstream_runoff[:2]
            ww, ww_profile = downstream_pws[:2]

            # Water-use rasters are NaN wherever there's no use, which means zero, not
            # missing data. Fill them with 0 inside the runoff footprint so WW is
            # averaged over the same cells as Q, and wc doesn't mask out cells.
            ww = numpy.where(numpy.isfinite(qd) & numpy.isnan(ww), 0.0, ww)
            wc = numpy.where(numpy.isfinite(qi) & numpy.isnan(wc), 0.0, wc)

            Q_downstream_avg = utilities.safe_mean(qd, qd_profile["nodata"])
            WW_downstream_avg = utilities.safe_mean(ww, ww_profile["nodata"])

            scalar_numer = Q_downstream_avg - WW_downstream_avg

            denominator = qi - wc
            denominator[numpy.abs(denominator) < 1e-6] = numpy.nan

            wrc = scalar_numer / denominator

            print(f"  WRC ({climate_model}, {year}):")
            print(f"  Q_downstream_avg:  {Q_downstream_avg:.3f} mm/yr")
            print(f"  WW_downstream_avg: {WW_downstream_avg:.3f} mm/yr")
            print(f"  Scalar numerator:  {scalar_numer:.3f} mm/yr")
            print(f"  Denominator range: {numpy.nanmin(denominator):.3f} to {numpy.nanmax(denominator):.3f} mm/yr")
            print(
                f"  WRC range (pre-mask): {numpy.nanmin(wrc):.3f} to {numpy.nanmax(wrc):.3f}"
            )  # CRB values are naturally extreme due to insufficient river contribution

            # Mask nodata cells
            nodata_mask = ~utilities.valid_mask(qi, qi_profile["nodata"]) | ~utilities.valid_mask(
                wc, wc_profile["nodata"]
            )
            wrc[nodata_mask] = numpy.nan
            wrc[~numpy.isfinite(wrc)] = numpy.nan

            wrc, profile, _ = utilities.save_raster(
                wrc,
                qi_profile,
                path.join(output_directory, f"{climate_model}_{year}_wrc.tif"),
            )
            result_arrays.append(wrc)

            plot_wrc_classes(
                wrc,
                profile,
                basin_hucs,
                path.join(output_directory, f"{climate_model}_{year}_wrc.png"),
                title=f"WRC ({climate_model}): {year}",
            )

            # TODO: stats and histogram

        assert profile is not None

        if len(YEARS) < 2:
            continue

        diff, profile, _ = utilities.compute_diff(
            result_arrays[0],
            result_arrays[-1],
            profile,
            path.join(output_directory, f"{climate_model}_diff_wrc.tif"),
        )

        finite = diff[numpy.isfinite(diff)]
        vmin, vmax = numpy.nanpercentile(finite, [2, 98])

        utilities.plot_raster_map(
            diff,
            profile,
            basin_hucs,
            path.join(output_directory, f"{climate_model}_diff_wrc.png"),
            title=f"WRC ({climate_model}): 2015-2020",
            cmap=plt.get_cmap("viridis").copy(),
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
            cbar_label="Water Resource Contribution (WRC)",
        )


def main():
    for basin, (
        basin_huc2s,
        excluded_upstream_huc4s,
        included_downstream_huc4s,
    ) in BASIN_MAPPINGS.items():
        if basin not in CONUS_BASINS:
            continue

        basin_hucs = pandas.concat(
            [geopandas.read_file(path.join("input", "HUC", huc2, "Shape", "WBDHU4.shp")) for huc2 in basin_huc2s]
        )
        basin_hucs.columns = [column.lower() for column in basin_hucs.columns]

        upstream_hucs = basin_hucs[~basin_hucs["huc4"].isin(excluded_upstream_huc4s)]
        downstream_hucs = basin_hucs[basin_hucs["huc4"].isin(included_downstream_huc4s)]

        output_directory = utilities.create_directory(path.join("output", basin), exist_ok=True)

        run_rwy(
            upstream_hucs,
            downstream_hucs,
            basin_hucs,
            output_directory,
        )

        run_wrc(upstream_hucs, downstream_hucs, basin_hucs, output_directory)

        #  Acknowledge limitation/improvement of proportional allocation from diverging HUC flows
        #  Acknowledge limitation: inter-basin transfers aren't modeled (see wrd_flows docstring)
        wrd_flows.run(basin_huc2s, basin_hucs, CLIMATE_MODELS, YEARS, output_directory)


if __name__ == "__main__":
    main()

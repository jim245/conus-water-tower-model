"""
CONUS-wide RWY / WRC / WRD maps, mosaicked from each basin's own outputs in
output/<basin>/ (run main.py first).

Every basin raster is reprojected onto the national pws-tot grid (EPSG:5070, 1 km)
— the grid WRC/WRD are already on, so those land cell-for-cell; RWY is on
Quickflow's 800 m grid and is nearest-neighbour resampled. main.CONUS_BASINS don't
overlap, so a cell takes the value of whichever basin covers it (boundary cells
can't double up: every clip uses the cell-centre rule).

Values aren't recomputed: each cell keeps what its own basin computed, i.e. relative
to that basin's outlet.
"""

from os import path
from typing import Any

import geopandas
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import pandas
import rasterio
from rasterio import enums, warp

import main
import network_indices
import utilities
import wrd_flows

OUTPUT_DIRECTORY = path.join("output", "CONUS")


def load_conus_hucs() -> geopandas.GeoDataFrame:
    huc2s = sorted({huc2 for basin in main.CONUS_BASINS for huc2 in main.BASIN_MAPPINGS[basin][0]})
    hucs = pandas.concat(
        [geopandas.read_file(path.join("input", "HUC", huc2, "Shape", "WBDHU4.shp")) for huc2 in huc2s]
    )
    hucs.columns = [column.lower() for column in hucs.columns]

    # Full-resolution WBD outlines take ~100 s per map to union and draw at CONUS
    # scale. Simplifying to one raster cell is invisible at this zoom; the coverage
    # variant keeps shared huc4 edges shared, so no slivers open up between them.
    hucs = hucs.to_crs("EPSG:5070")
    hucs["geometry"] = hucs.geometry.simplify_coverage(tolerance=1000)
    return hucs


def mosaic(raster_paths: list[str], reference_profile: dict[str, Any]) -> numpy.ndarray:
    """Reproject each raster onto the reference grid and composite them, first
    non-NaN value wins."""
    height, width = reference_profile["height"], reference_profile["width"]
    result = numpy.full((height, width), numpy.nan, dtype="float32")

    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            destination = numpy.full((height, width), numpy.nan, dtype="float32")
            warp.reproject(
                src.read(1).astype("float32"),
                destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=numpy.nan,
                dst_transform=reference_profile["transform"],
                dst_crs=reference_profile["crs"],
                dst_nodata=numpy.nan,
                resampling=enums.Resampling.nearest,
            )
        result = numpy.where(numpy.isnan(result), destination, result)

    return result


def basin_paths(pattern: str) -> list[str]:
    """`pattern` (relative to output/<basin>/) for every CONUS basin, warning about
    basins that don't have it yet."""
    paths = []
    for basin in main.CONUS_BASINS:
        basin_path = path.join("output", basin, pattern)
        if path.exists(basin_path):
            paths.append(basin_path)
        else:
            print(f"  WARNING: {basin_path} missing — {basin} will be blank")
    return paths


def plot_rwy(
    rwy: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: str,
    title: str,
):
    """Continuous RWY map, for the 2015-2020 diff: the RWY classes are for RWY
    itself, not a change in it."""
    finite = rwy[numpy.isfinite(rwy)]
    vmin, vmax = numpy.nanpercentile(finite, [2, 98])

    utilities.plot_raster_map(
        rwy,
        profile,
        hucs,
        output_path,
        title=title,
        cmap=plt.get_cmap("YlOrRd").copy(),
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        cbar_label="Runoff Water Yield (RWY)",
    )


def run():
    utilities.create_directory(OUTPUT_DIRECTORY)
    hucs = load_conus_hucs()
    reference_profile = utilities.read_raster_profile(
        path.join("input", "pws-tot", f"pws-tot_huc12_{main.YEARS[0]}.tif")
    )
    reference_profile.update(compress="lzw")

    for climate_model in main.CLIMATE_MODELS:
        rwy_directory = utilities.create_directory(path.join(OUTPUT_DIRECTORY, "rwy", climate_model))
        wrc_directory = utilities.create_directory(path.join(OUTPUT_DIRECTORY, "wrc", climate_model))

        for year in [*main.YEARS, *(["diff"] if len(main.YEARS) > 1 else [])]:
            print(f"CONUS {climate_model} {year}")

            rwy = mosaic(
                basin_paths(path.join("rwy", climate_model, f"{climate_model}_{year}_rwy.tif")),
                reference_profile,
            )
            rwy, profile, _ = utilities.save_raster(
                rwy,
                reference_profile,
                path.join(rwy_directory, f"{climate_model}_{year}_rwy.tif"),
            )
            rwy_path = path.join(rwy_directory, f"{climate_model}_{year}_rwy.png")
            if year == "diff":
                plot_rwy(rwy, profile, hucs, rwy_path, f"RWY ({climate_model}): 2015-2020")
            else:
                main.plot_rwy_classes(rwy, profile, hucs, rwy_path, f"RWY ({climate_model}): {year}")

            wrc = mosaic(
                basin_paths(path.join("wrc", climate_model, f"{climate_model}_{year}_wrc.tif")),
                reference_profile,
            )
            wrc, profile, _ = utilities.save_raster(
                wrc,
                reference_profile,
                path.join(wrc_directory, f"{climate_model}_{year}_wrc.tif"),
            )
            if year == "diff":
                # Same continuous viridis map run_wrc uses for its diff.
                finite = wrc[numpy.isfinite(wrc)]
                vmin, vmax = numpy.nanpercentile(finite, [2, 98])
                utilities.plot_raster_map(
                    wrc,
                    profile,
                    hucs,
                    path.join(wrc_directory, f"{climate_model}_diff_wrc.png"),
                    title=f"WRC ({climate_model}): 2015-2020",
                    cmap=plt.get_cmap("viridis").copy(),
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                    cbar_label="Water Resource Contribution (WRC)",
                )
                continue  # WRD has no diff

            main.plot_wrc_classes(
                wrc,
                profile,
                hucs,
                path.join(wrc_directory, f"{climate_model}_{year}_wrc.png"),
                title=f"WRC ({climate_model}): {year}",
            )

            for level in wrd_flows.LEVELS:
                run_wrd(hucs, reference_profile, climate_model, year, level)
            for level in network_indices.LEVELS:
                run_network_rwy_wrc(hucs, reference_profile, climate_model, year, level)

    run_diffs(hucs)


# (index, cbar label, cmap) for each per-level product without a basin-level diff.
# RWY/WRC keep the colormaps of their main.py diffs; WRD changes sign, so it gets a
# diverging map centred on 0.
DIFF_INDICES = [
    ("rwy", "Runoff Water Yield (RWY)", "YlOrRd", network_indices.LEVELS),
    ("wrc", "Water Resource Contribution (WRC)", "viridis", network_indices.LEVELS),
    ("wrd", "Water Resource Dependency (WRD)", "RdBu_r", wrd_flows.LEVELS),
]


def run_diffs(hucs: geopandas.GeoDataFrame) -> None:
    """Last-minus-first-year diff of the CONUS network RWY/WRC and WRD mosaics. The
    basin runs don't write diffs at these levels, so they're taken from the CONUS
    mosaics directly; every year shares the reference grid, so they align
    cell-for-cell."""
    if len(main.YEARS) < 2:
        return
    first, last = main.YEARS[0], main.YEARS[-1]

    for climate_model in main.CLIMATE_MODELS:
        for index, cbar_label, cmap, levels in DIFF_INDICES:
            for level in levels:
                directory = path.join(OUTPUT_DIRECTORY, index, f"huc{level}", climate_model)
                print(f"CONUS {index.upper()} huc{level} {climate_model} diff")
                starting, _, profile = utilities.read_raster(
                    path.join(directory, f"{climate_model}_{first}_{index}.tif")
                )
                ending, _, _ = utilities.read_raster(path.join(directory, f"{climate_model}_{last}_{index}.tif"))
                diff, profile, _ = utilities.compute_diff(
                    starting,
                    ending,
                    profile,
                    path.join(directory, f"{climate_model}_diff_{index}.tif"),
                )

                finite = diff[numpy.isfinite(diff)]
                vmin, vmax = numpy.nanpercentile(finite, [2, 98])
                if index == "wrd":
                    # Symmetric about 0 so equal gains and losses get equal color.
                    limit = max(abs(vmin), abs(vmax))
                    norm = mcolors.Normalize(vmin=-limit, vmax=limit)
                else:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                utilities.plot_raster_map(
                    diff,
                    profile,
                    hucs,
                    path.join(directory, f"{climate_model}_diff_{index}.png"),
                    title=f"{index.upper()} huc{level} ({climate_model}): {first}-{last}",
                    cmap=plt.get_cmap(cmap).copy(),
                    norm=norm,
                    cbar_label=f"Δ {cbar_label}",
                )


def run_network_rwy_wrc(
    hucs: geopandas.GeoDataFrame,
    reference_profile: dict[str, Any],
    climate_model: str,
    year: int,
    level: int,
) -> None:
    """Mosaic network_indices' RWY and WRC at one level."""
    level_name = f"huc{level}"
    for index, plot_classes in [
        ("rwy", main.plot_rwy_classes),
        ("wrc", main.plot_wrc_classes),
    ]:
        directory = utilities.create_directory(path.join(OUTPUT_DIRECTORY, index, level_name, climate_model))
        data = mosaic(
            basin_paths(path.join(index, level_name, climate_model, f"{climate_model}_{year}_{index}.tif")),
            reference_profile,
        )
        data, profile, _ = utilities.save_raster(
            data,
            reference_profile,
            path.join(directory, f"{climate_model}_{year}_{index}.tif"),
        )
        plot_classes(
            data,
            profile,
            hucs,
            path.join(directory, f"{climate_model}_{year}_{index}.png"),
            title=f"{index.upper()} {level_name} ({climate_model}): {year}",
        )


def run_wrd(
    hucs: geopandas.GeoDataFrame,
    reference_profile: dict[str, Any],
    climate_model: str,
    year: int,
    level: int,
) -> None:
    wrd_directory = utilities.create_directory(path.join(OUTPUT_DIRECTORY, "wrd", f"huc{level}", climate_model))
    wrd = mosaic(
        basin_paths(path.join("wrd", f"huc{level}", climate_model, f"{climate_model}_{year}_wrd.tif")),
        reference_profile,
    )
    wrd, profile, _ = utilities.save_raster(
        wrd,
        reference_profile,
        path.join(wrd_directory, f"{climate_model}_{year}_wrd.tif"),
    )
    wrd_flows.plot_wrd_cells(
        wrd,
        profile,
        hucs,
        path.join(wrd_directory, f"{climate_model}_{year}_wrd.png"),
        f"WRD huc{level} ({climate_model}): {year}",
    )


if __name__ == "__main__":
    run()

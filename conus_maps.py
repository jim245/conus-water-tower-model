"""
Makes the CONUS maps of RWY, WRC and WRD from the basin outputs in output/<basin>/ (main.py). Writes output/CONUS/.

Each map is a mosaic of the basin rasters of main.CONUS_BASINS. RWY is on the grid of the national mean Quickflow
(raster_preparation.national_quickflow_path). WRC and WRD are on the 1 km pws-tot grid. The basin rasters are clips of
these grids, thus the mosaic copies each basin value to its cell. The basins do not overlap, and each clip uses the
cell-centre rule, thus each cell has the value of one basin. The values are not computed again: each cell keeps the
value that its basin computed.

Blank areas: huc2 09 is in no basin. At the basin level, RWY and WRC are for the upstream cells only, thus the excluded
and outlet huc4s of main.BASIN_MAPPINGS are blank. A basin that does not have a file is blank in that map.

Reads the files that are in output/<basin>/. Does not check that they come from the current code and YEARS.
"""

import os

# the reason is in main.py. Set it before the imports that load PROJ
os.environ["PROJ_NETWORK"] = "OFF"

# isort: split

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import pandas
import rasterio
import shapely
from rasterio import enums, warp

import index_maps
import main
import raster_preparation
import utilities
import wrd_flows

OUTPUT_DIRECTORY = Path("output", "CONUS")

OUTLINE_TOLERANCE = 1000  # m, one cell of the 1 km grid
OUTLINE_MIN_HOLE_AREA = 1e6  # m², one cell of the 1 km grid

# (index, colorbar label, colormap, True if the index has a basin-level map). RWY and WRC use the colormaps of the
# main.py diffs
DIFF_INDICES = [
    ("rwy", "Relative Water Yield (RWY)", "YlOrRd", True),
    ("wrc", "Water Resource Contribution (WRC)", "viridis", True),
    ("wrd", "Water Resource Dependency (WRD)", "RdBu_r", False),
]


def load_conus_hucs() -> geopandas.GeoDataFrame:
    """
    Makes the huc4 polygons of all basins of main.CONUS_BASINS, for the map outlines.

    The polygons are the huc4 polygons of the basin stages: the GDB huc12s dissolved by code (wrd_flows.load_units).
    Simplifies each polygon by OUTLINE_TOLERANCE. Thus the two sides of a shared edge can be up to OUTLINE_TOLERANCE
    apart, less than one pixel of a CONUS map. Then removes the holes smaller than OUTLINE_MIN_HOLE_AREA (gaps between
    the huc12s of a huc4). The polygons only draw the outlines: no raster value depends on them.

    Returns:
        hucs (geopandas.GeoDataFrame): One row for each huc4, with the columns unit and geometry, in EPSG:5070.
    """

    huc2s = sorted({huc2 for basin in main.CONUS_BASINS for huc2 in main.BASIN_MAPPINGS[basin][0]})
    # one huc2 at a time, so that the huc12s of only one huc2 are in memory
    hucs = pandas.concat(
        [wrd_flows.load_units(wrd_flows.load_huc12s([huc2]), 4) for huc2 in huc2s], ignore_index=True
    ).to_crs("EPSG:5070")

    # full-resolution outlines make each CONUS map slow to draw. The GDB huc4s are not a valid coverage, and
    # simplify_coverage fails on them
    hucs["geometry"] = hucs.geometry.simplify(OUTLINE_TOLERANCE, preserve_topology=True)
    hucs["geometry"] = geopandas.GeoSeries(
        [remove_small_holes(geometry, OUTLINE_MIN_HOLE_AREA) for geometry in hucs.geometry],
        index=hucs.index,
        crs=hucs.crs,
    )
    return hucs


def remove_small_holes(geometry: shapely.Geometry, min_area: float) -> shapely.Geometry:
    """
    Removes the holes of a polygon or multipolygon that have an area less than min_area.

    Parameters:
        geometry (shapely.Geometry): A polygon or multipolygon.
        min_area (float): The smallest hole area to keep, in the units of the CRS squared.

    Returns:
        geometry (shapely.Geometry): The geometry without the small holes, a multipolygon if it has more than one part.
    """

    polygons = [
        shapely.Polygon(
            polygon.exterior, [ring for ring in polygon.interiors if shapely.Polygon(ring).area >= min_area]
        )
        for polygon in shapely.get_parts(geometry)
    ]
    return shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]


def mosaic(raster_paths: Sequence[Path], profile: dict[str, Any]) -> numpy.ndarray:
    """
    Reprojects each raster to the grid of profile with nearest resampling, and combines them.

    Where two rasters have a value, the first raster in raster_paths gives the value.

    Parameters:
        raster_paths (Sequence[Path]): The rasters.
        profile (dict[str, Any]): The profile of the output grid.

    Returns:
        mosaic (numpy.ndarray): The (height, width) float32 mosaic, NaN where no raster has a value.
    """

    height, width = profile["height"], profile["width"]
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
                dst_transform=profile["transform"],
                dst_crs=profile["crs"],
                dst_nodata=numpy.nan,
                resampling=enums.Resampling.nearest,
            )
        result = numpy.where(numpy.isnan(result), destination, result)

    return result


def basin_paths(relative_path: Path, missing: list[Path]) -> list[Path]:
    """
    Finds a basin output file in each basin of main.CONUS_BASINS.

    Prints a warning for each basin that does not have the file, and adds its path to missing.

    Parameters:
        relative_path (Path): The path of the file under output/<basin>/.
        missing (list[Path]): The paths that do not exist. This function adds to it.

    Returns:
        paths (list[Path]): The paths that exist, in the order of main.CONUS_BASINS.
    """

    paths = list[Path]()
    for basin in main.CONUS_BASINS:
        basin_path = Path("output", basin, relative_path)
        if basin_path.exists():
            paths.append(basin_path)
        else:
            missing.append(basin_path)
            print(f"  WARNING: {basin_path} missing — {basin} will be blank")
    return paths


def run() -> None:
    """
    Makes the CONUS maps of each scenario and year, and the diffs (run_diffs).

    The maps are RWY and WRC at the basin level, WRD at each level (run_wrd), and RWY and WRC at each network level
    (run_network_rwy_wrc). Prints the number of missing basin files at the end.

    Writes <scenario>_<year>_rwy.tif and .png in rwy/<scenario>/, and <scenario>_<year>_wrc.tif and .png in
    wrc/<scenario>/, under OUTPUT_DIRECTORY (and the files of run_wrd, run_network_rwy_wrc and run_diffs).
    """

    _ = utilities.create_directory(OUTPUT_DIRECTORY)
    hucs = load_conus_hucs()
    reference_profile = utilities.read_raster_profile(raster_preparation.water_use_grid_path(main.YEARS[0]))
    reference_profile.update(compress="lzw")
    rwy_profile = utilities.read_raster_profile(
        raster_preparation.national_quickflow_path(main.CLIMATE_MODELS[0], main.YEARS[0])
    )
    rwy_profile.update(compress="lzw")
    missing = list[Path]()

    for climate_model in main.CLIMATE_MODELS:
        rwy_directory = utilities.create_directory(Path(OUTPUT_DIRECTORY, "rwy", climate_model))
        wrc_directory = utilities.create_directory(Path(OUTPUT_DIRECTORY, "wrc", climate_model))

        for year in main.YEARS:
            print(f"CONUS {climate_model} {year}")

            rwy = mosaic(
                basin_paths(Path("rwy", climate_model, f"{climate_model}_{year}_rwy.tif"), missing), rwy_profile
            )
            rwy, profile, _ = utilities.save_raster(
                rwy, rwy_profile, Path(rwy_directory, f"{climate_model}_{year}_rwy.tif")
            )
            index_maps.plot_rwy_classes(
                rwy,
                profile,
                hucs,
                Path(rwy_directory, f"{climate_model}_{year}_rwy.png"),
                f"RWY ({climate_model}): {year}",
            )

            wrc = mosaic(
                basin_paths(Path("wrc", climate_model, f"{climate_model}_{year}_wrc.tif"), missing), reference_profile
            )
            wrc, profile, _ = utilities.save_raster(
                wrc, reference_profile, Path(wrc_directory, f"{climate_model}_{year}_wrc.tif")
            )
            index_maps.plot_wrc_classes(
                wrc,
                profile,
                hucs,
                Path(wrc_directory, f"{climate_model}_{year}_wrc.png"),
                title=f"WRC ({climate_model}): {year}",
            )

            for level in wrd_flows.LEVELS:
                run_wrd(hucs, reference_profile, climate_model, year, level, missing)
            for level in wrd_flows.LEVELS:
                run_network_rwy_wrc(hucs, rwy_profile, reference_profile, climate_model, year, level, missing)

    run_diffs(hucs)

    if missing:
        basins = sorted({basin_path.parts[1] for basin_path in missing})
        print(f"WARNING: {len(missing)} basin files missing, from the basins {basins}: these are blank in their maps")


def run_diffs(hucs: geopandas.GeoDataFrame) -> None:
    """
    Writes the diff of the last and the first year of each CONUS mosaic, and its map.

    The diffs are of RWY and WRC at the basin level, and of RWY, WRC and WRD at each network level (DIFF_INDICES). The
    years are on one grid, thus the diff is cell by cell. The color scale is from the 2nd to the 98th percentile, and
    symmetric about 0 for WRD. A diff with no value gets no map: the function prints a warning. Does nothing if YEARS
    has fewer than 2 years.

    Writes <scenario>_diff_<index>.tif and .png in the folder of the year files.

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons, for the map outlines.
    """

    if len(main.YEARS) < 2:
        return
    first, last = main.YEARS[0], main.YEARS[-1]

    for climate_model in main.CLIMATE_MODELS:
        for index, cbar_label, cmap, has_basin_level in DIFF_INDICES:
            levels: list[int | None] = [*([None] if has_basin_level else []), *wrd_flows.LEVELS]
            for level in levels:
                if level is None:
                    directory = Path(OUTPUT_DIRECTORY, index, climate_model)
                    name = f"{index.upper()} ({climate_model})"
                    label = cbar_label
                else:
                    directory = Path(OUTPUT_DIRECTORY, index, f"huc{level}", climate_model)
                    name = f"{index.upper()} huc{level} ({climate_model})"
                    label = f"Δ {cbar_label}"
                print(f"CONUS {name} diff")
                starting, _, profile = utilities.read_raster(Path(directory, f"{climate_model}_{first}_{index}.tif"))
                ending, _, _ = utilities.read_raster(Path(directory, f"{climate_model}_{last}_{index}.tif"))
                diff, profile, _ = utilities.compute_diff(
                    starting, ending, profile, Path(directory, f"{climate_model}_diff_{index}.tif")
                )

                finite = diff[numpy.isfinite(diff)]
                if finite.size == 0:
                    print(f"  WARNING: {name} diff has no value, no diff map")
                    continue
                vmin, vmax = numpy.nanpercentile(finite, [2, 98])
                if index == "wrd":
                    limit = max(abs(vmin), abs(vmax))
                    norm = mcolors.Normalize(vmin=-limit, vmax=limit)
                else:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                utilities.plot_raster_map(
                    diff,
                    profile,
                    hucs,
                    Path(directory, f"{climate_model}_diff_{index}.png"),
                    title=f"{name}: {first}-{last}",
                    cmap=plt.get_cmap(cmap).copy(),
                    norm=norm,
                    cbar_label=label,
                )


def run_network_rwy_wrc(
    hucs: geopandas.GeoDataFrame,
    rwy_profile: dict[str, Any],
    reference_profile: dict[str, Any],
    climate_model: str,
    year: int,
    level: int,
    missing: list[Path],
) -> None:
    """
    Mosaics the RWY and the WRC of network_indices at one level, for one scenario and year, and plots the class maps.

    Writes <scenario>_<year>_rwy.tif and .png in rwy/huc<level>/<scenario>/, and <scenario>_<year>_wrc.tif and .png in
    wrc/huc<level>/<scenario>/, under OUTPUT_DIRECTORY.

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons, for the map outlines.
        rwy_profile (dict[str, Any]): The profile of the RWY grid.
        reference_profile (dict[str, Any]): The profile of the WRC grid.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year.
        level (int): The level, for example 8.
        missing (list[Path]): The missing basin files (basin_paths). This function adds to it.
    """

    level_name = f"huc{level}"
    for index, plot_classes, grid_profile in [
        ("rwy", index_maps.plot_rwy_classes, rwy_profile),
        ("wrc", index_maps.plot_wrc_classes, reference_profile),
    ]:
        directory = utilities.create_directory(Path(OUTPUT_DIRECTORY, index, level_name, climate_model))
        data = mosaic(
            basin_paths(Path(index, level_name, climate_model, f"{climate_model}_{year}_{index}.tif"), missing),
            grid_profile,
        )
        data, profile, _ = utilities.save_raster(
            data, grid_profile, Path(directory, f"{climate_model}_{year}_{index}.tif")
        )
        plot_classes(
            data,
            profile,
            hucs,
            Path(directory, f"{climate_model}_{year}_{index}.png"),
            title=f"{index.upper()} {level_name} ({climate_model}): {year}",
        )


def run_wrd(
    hucs: geopandas.GeoDataFrame,
    reference_profile: dict[str, Any],
    climate_model: str,
    year: int,
    level: int,
    missing: list[Path],
) -> None:
    """
    Mosaics the WRD of wrd_flows at one level, for one scenario and year, and plots the class map.

    Writes <scenario>_<year>_wrd.tif and .png in wrd/huc<level>/<scenario>/ under OUTPUT_DIRECTORY.

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons, for the map outlines.
        reference_profile (dict[str, Any]): The profile of the WRD grid.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year.
        level (int): The level, for example 8.
        missing (list[Path]): The missing basin files (basin_paths). This function adds to it.
    """

    wrd_directory = utilities.create_directory(Path(OUTPUT_DIRECTORY, "wrd", f"huc{level}", climate_model))
    wrd = mosaic(
        basin_paths(Path("wrd", f"huc{level}", climate_model, f"{climate_model}_{year}_wrd.tif"), missing),
        reference_profile,
    )
    wrd, profile, _ = utilities.save_raster(
        wrd, reference_profile, Path(wrd_directory, f"{climate_model}_{year}_wrd.tif")
    )
    wrd_flows.plot_wrd_cells(
        wrd,
        profile,
        hucs,
        Path(wrd_directory, f"{climate_model}_{year}_wrd.png"),
        f"WRD huc{level} ({climate_model}): {year}",
    )


if __name__ == "__main__":
    run()

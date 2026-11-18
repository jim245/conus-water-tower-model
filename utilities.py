"""
Raster helpers that the pipeline modules share: read, save, clip, reproject, sum, unit conversion and maps.

Each helper that writes a raster calls save_raster, which writes single-band float32 with NaN as nodata.
"""

import calendar
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas
import matplotlib.pyplot as plt
import numpy
import rasterio
from matplotlib.colors import Colormap, Normalize
from numpy.typing import NDArray
from rasterio import enums, io, mask, transform, warp
from rasterio.crs import CRS

RasterResult = tuple[numpy.ndarray, dict[str, Any], Path]  # (data, profile, path)

CUBIC_METRES_PER_MILLION_GALLONS = 3785.411784
MM_PER_M = 1000

WRC_MIN_DENOMINATOR = 1e-6  # mm/yr. A cell with a smaller |qi - wc| has no WRC, as in the original method


def create_directory(directory: Path) -> Path:
    """
    Creates a directory and its missing parent directories, if they do not exist.

    Parameters:
        directory (Path): The directory path to create.

    Returns:
        directory (Path): The directory path.
    """

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compute_stats(arr: numpy.ndarray, label: str) -> dict[str, float]:
    """
    Computes the min, quartiles, median, mean and max of the finite cells, and prints them.

    Parameters:
        arr (numpy.ndarray): The raster data.
        label (str): The name of the data, for the printed line and the error message.

    Returns:
        stats (dict[str, float]): The statistics, by the keys min, q1, median, mean, q3 and max.

    Raises:
        ValueError: If no cell is finite.
    """

    valid = arr[numpy.isfinite(arr)]
    if valid.size == 0:
        raise ValueError(f"{label} has no valid data for statistics")

    stats = {
        "min": float(numpy.min(valid)),
        "q1": float(numpy.percentile(valid, 25)),
        "median": float(numpy.median(valid)),
        "mean": float(numpy.mean(valid)),
        "q3": float(numpy.percentile(valid, 75)),
        "max": float(numpy.max(valid)),
    }
    print(
        f"  {label} stats: min={stats['min']:.3f}  q1={stats['q1']:.3f}  "
        f"median={stats['median']:.3f}  mean={stats['mean']:.3f}  "
        f"q3={stats['q3']:.3f}  max={stats['max']:.3f}"
    )
    return stats


def print_stats(data: numpy.ndarray, label: str) -> None:
    """
    Prints the statistics of the finite cells (compute_stats), or a warning if no cell is finite.

    Parameters:
        data (numpy.ndarray): The raster data.
        label (str): The name of the data, for the printed line.
    """

    if numpy.isfinite(data).any():
        _ = compute_stats(data, label)
    else:
        print(f"  WARNING: {label} has no value, no statistics")


def compute_diff(
    starting_data: numpy.ndarray,
    ending_data: numpy.ndarray,
    profile: dict[str, Any],
    output_path: Path,
) -> RasterResult:
    """
    Writes ending_data - starting_data with save_raster.

    A cell that is NaN in one of the two inputs is NaN in the diff.

    Parameters:
        starting_data (numpy.ndarray): The raster of the first year.
        ending_data (numpy.ndarray): The raster of the last year, on the same grid.
        profile (dict[str, Any]): The profile of the grid.
        output_path (Path): The path to the output file.

    Returns:
        result (RasterResult): The save_raster result.
    """

    result = save_raster(ending_data - starting_data, profile, output_path)
    print(f"  diff saved → {output_path}")
    return result


def valid_mask(arr: NDArray[numpy.floating], nodata: float | None) -> NDArray[numpy.bool]:
    """
    Finds the cells that are finite and not nodata.

    Parameters:
        arr (NDArray[numpy.floating]): The raster data.
        nodata (float | None): The nodata value. None or NaN when only the non-finite cells are invalid.

    Returns:
        valid (NDArray[numpy.bool]): True for the valid cells.
    """

    if nodata is None:
        return numpy.isfinite(arr)

    return numpy.isfinite(arr) & (arr != nodata)


def safe_mean(arr: NDArray[numpy.floating], nodata: float | None) -> float:
    """
    Computes the mean of the finite, non-negative cells that are not nodata.

    Parameters:
        arr (NDArray[numpy.floating]): The raster data.
        nodata (float | None): The nodata value (see valid_mask).

    Returns:
        mean (float): The mean.

    Raises:
        ValueError: If no cell is valid.
    """

    valid = arr[valid_mask(arr, nodata) & (arr >= 0)]
    if valid.size == 0:
        raise ValueError("no finite, non-negative cell for the mean")

    return float(numpy.mean(valid))


def clip_raster(
    data: numpy.ndarray,
    profile: dict[str, Any],
    raster_mask: geopandas.GeoSeries,
    output_path: Path,
) -> RasterResult:
    """
    Clips a raster to the mask polygons, crops it to their bounds, and writes it with save_raster.

    Cells outside the polygons are NaN.

    Parameters:
        data (numpy.ndarray): The (height, width) raster data.
        profile (dict[str, Any]): The profile of the raster.
        raster_mask (geopandas.GeoSeries): The polygons to clip to, in any CRS.
        output_path (Path): The path to the output file.

    Returns:
        result (RasterResult): The save_raster result.
    """

    with io.MemoryFile() as memfile, memfile.open(**profile) as dataset:
        dataset.write(data, 1)
        # all_touched=False: a cell belongs to the polygon that contains its centre. Thus a cell on a shared boundary
        # (for example upstream/downstream) goes to one side only
        clipped, clipped_transform = mask.mask(
            dataset,
            raster_mask.to_crs(dataset.crs).geometry,
            all_touched=False,
            crop=True,
            nodata=numpy.nan,
        )

    clipped_profile = dict(profile)
    clipped_profile.update(height=clipped.shape[1], transform=clipped_transform, width=clipped.shape[2])

    return save_raster(clipped[0], clipped_profile, output_path)


def reproject_raster(
    data: numpy.ndarray,
    source_profile: dict[str, Any],
    reference_profile: dict[str, Any],
    output_path: Path,
    *,
    resampling_method: enums.Resampling,
    scale_factor: float = 1.0,
) -> RasterResult:
    """
    Reprojects and resamples a raster onto the grid of the reference profile, and writes it with save_raster.

    Cells that are nodata in the source do not contribute.

    Parameters:
        data (numpy.ndarray): The (height, width) raster data.
        source_profile (dict[str, Any]): The profile of the raster.
        reference_profile (dict[str, Any]): The profile whose CRS, transform and size the output gets.
        output_path (Path): The path to the output file.
        resampling_method (enums.Resampling): The resampling method. Use sum for a total per cell, and average for a
            value per area.
        scale_factor (float = 1.0): The factor for each output cell, for example for a unit conversion.

    Returns:
        result (RasterResult): The save_raster result.
    """

    source = numpy.where(valid_mask(data, source_profile.get("nodata")), data, numpy.nan).astype(numpy.float32)

    profile = dict(source_profile)
    profile.update(
        compress="lzw",
        driver="GTiff",
        crs=reference_profile["crs"],
        transform=reference_profile["transform"],
        width=reference_profile["width"],
        height=reference_profile["height"],
    )

    destination = numpy.full((profile["height"], profile["width"]), numpy.nan, dtype=numpy.float32)
    # without src_nodata, one nodata source cell makes the whole target cell nodata under sum and average. Without
    # dst_nodata, target cells outside the source get 0, not nodata
    warp.reproject(
        source,
        destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        src_nodata=numpy.nan,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        dst_nodata=numpy.nan,
        resampling=resampling_method,
    )

    return save_raster(destination * scale_factor, profile, output_path)


def convert_mgd_to_mm_per_year(
    data: numpy.ndarray,
    nodata: float | None,
    profile: dict[str, Any],
    year: int,
    output_path: Path,
) -> RasterResult:
    """
    Converts a raster of annual mean MGD to mm/yr over each cell, and writes it with save_raster.

    The cell area comes from the transform. Thus the CRS must be equal-area and in metres (for example EPSG:5070).

    Parameters:
        data (numpy.ndarray): The (height, width) water use, in MGD.
        nodata (float | None): The nodata value (see valid_mask).
        profile (dict[str, Any]): The profile of the raster.
        year (int): The year of the data, for the number of days.
        output_path (Path): The path to the output file.

    Returns:
        result (RasterResult): The save_raster result.

    Raises:
        ValueError: If the CRS is not projected in metres.
    """

    crs = CRS.from_user_input(profile["crs"])
    if not crs.is_projected or crs.linear_units != "metre":
        raise ValueError(f"MGD to mm/yr needs a projected CRS in metres, not {crs}")

    cell_area_m2 = abs(profile["transform"].a * profile["transform"].e)
    days = 366 if calendar.isleap(year) else 365
    conversion_factor = days * CUBIC_METRES_PER_MILLION_GALLONS * MM_PER_M / cell_area_m2
    result = numpy.where(valid_mask(data, nodata), data * conversion_factor, numpy.nan).astype(numpy.float32)

    return save_raster(result, profile, output_path)


def read_raster(path: Path) -> tuple[numpy.ndarray, float | None, dict[str, Any]]:
    """
    Reads the first band of a raster.

    The return order is not the RasterResult order.

    Parameters:
        path (Path): The path to the raster.

    Returns:
        data (numpy.ndarray): The band, NaN in the nodata cells.
        nodata (float | None): The nodata value of the file.
        profile (dict[str, Any]): The profile of the file.
    """

    with rasterio.open(path) as raster:
        data = raster.read(1, masked=True).filled(numpy.nan)
        return data, raster.nodata, raster.profile


def read_raster_profile(path: Path) -> dict[str, Any]:
    """
    Reads the profile of a raster (CRS, transform, size and more), without the cell data.

    Parameters:
        path (Path): The path to the raster.

    Returns:
        profile (dict[str, Any]): The profile of the file.
    """

    with rasterio.open(path) as raster:
        return raster.profile


def save_raster(arr: numpy.ndarray, profile: dict[str, Any], path: Path) -> RasterResult:
    """
    Writes a single-band float32 GeoTIFF with NaN as nodata. Non-finite cells become NaN.

    The data goes to a temporary file in the same directory, which is then renamed to path. Thus an interrupted write
    can't leave a partial raster at path.

    Parameters:
        arr (numpy.ndarray): The (height, width) raster data.
        profile (dict[str, Any]): The profile of the raster. This function replaces its dtype and nodata.
        path (Path): The path to the output file.

    Returns:
        result (RasterResult): The written data, the written profile and path.
    """

    profile = dict(profile)
    profile.update(driver="GTiff", dtype=rasterio.float32, nodata=numpy.nan)
    out = numpy.where(numpy.isfinite(arr), arr, numpy.nan).astype(numpy.float32)

    # the process ID keeps concurrent runs from sharing a temporary file
    temporary_path = Path(f"{path}.{os.getpid()}.tmp")
    with rasterio.open(temporary_path, "w", **profile) as dst:
        dst.write(out, 1)
    temporary_path.replace(path)

    return out, profile, path


def sum_rasters(rasters: Sequence[RasterResult], output_path: Path) -> RasterResult:
    """
    Sums rasters cell by cell over their valid values, and writes the sum with save_raster.

    A cell that is NaN in all inputs stays NaN.

    Parameters:
        rasters (Sequence[RasterResult]): The rasters, all on the grid of the first one.
        output_path (Path): The path to the output file.

    Returns:
        result (RasterResult): The save_raster result, with the profile of the first raster.

    Raises:
        ValueError: If a raster is not on the grid of the first one.
    """

    arrays = [data for data, _, _ in rasters]
    _, first_profile, first_path = rasters[0]
    for arr, profile, raster_path in rasters:
        if arr.shape != arrays[0].shape or not profile["transform"].almost_equals(first_profile["transform"]):
            raise ValueError(f"{raster_path} isn't on the grid of {first_path}")

    stack = numpy.stack(arrays)
    result = numpy.nansum(stack, axis=0)
    result[numpy.all(numpy.isnan(stack), axis=0)] = numpy.nan

    raster_result = save_raster(result, first_profile, output_path)
    print(f"Sum raster saved → {output_path}")
    return raster_result


def plot_raster_map(
    data: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: Path,
    *,
    title: str,
    cmap: Colormap,
    norm: Normalize,
    cbar_label: str,
    cbar_ticks: Sequence[float] | None = None,
    cbar_ticklabels: Sequence[str] | None = None,
) -> None:
    """
    Plots a single-band raster with the HUC outlines and a colorbar, and saves it as a PNG.

    NaN cells are transparent.

    Parameters:
        data (numpy.ndarray): The (height, width) raster data, already classified or normalized.
        profile (dict[str, Any]): The profile of the raster.
        hucs (geopandas.GeoDataFrame): The HUCs whose outlines to plot, in any CRS.
        output_path (Path): The path to the output PNG.
        title (str): The map title.
        cmap (Colormap): The colormap. The caller's colormap does not change.
        norm (Normalize): The normalization from data to colormap.
        cbar_label (str): The colorbar label.
        cbar_ticks (Sequence[float] | None = None): The colorbar ticks. Matplotlib's ticks when None.
        cbar_ticklabels (Sequence[str] | None = None): The colorbar tick labels. The tick values when None.
    """

    cmap = cmap.with_extremes(bad=(0, 0, 0, 0))

    height, width = data.shape
    left, bottom, right, top = transform.array_bounds(height, width, profile["transform"])
    hucs_proj = hucs.to_crs(profile["crs"])

    # frame the map on the raster and on the HUC outlines, because the raster often covers only some of the HUCs (for
    # example upstream only). The margin prevents a cut through the edge outlines
    huc_left, huc_bottom, huc_right, huc_top = hucs_proj.total_bounds
    view_left, view_bottom = min(left, huc_left), min(bottom, huc_bottom)
    view_right, view_top = max(right, huc_right), max(top, huc_top)
    margin = 0.02 * max(view_right - view_left, view_top - view_bottom)

    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(
        data.astype("float32"),
        extent=(left, right, bottom, top),
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )
    # aspect=None prevents geopandas from calling ax.set_aspect
    hucs_proj.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=3, aspect=None)
    # set the limits after all plot calls, because each plot call can change them
    ax.set_xlim(view_left - margin, view_right + margin)
    ax.set_ylim(view_bottom - margin, view_top + margin)
    ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(img, ax=ax, ticks=cbar_ticks)
    if cbar_ticklabels is not None:
        cbar.ax.set_yticklabels(cbar_ticklabels)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Map saved → {output_path}")

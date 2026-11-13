import os
from typing import Any, Sequence

import contextily
import geopandas
import matplotlib.pyplot as plt
import numpy
import rasterio
from matplotlib.colors import Colormap, Normalize
from numpy.typing import NDArray
from rasterio import enums, io, mask, transform, warp
from shapely.geometry import box as shapely_box

RasterResult = tuple[numpy.ndarray, dict[str, Any], str]  # (data, profile, path)


def create_directory(directory: str, exist_ok: bool = True) -> str:
    """
    Creates a specified directory and all intermediate directories.

    Parameters:
        directory (str): The directory path to create.
        exist_ok (bool = True): Returns an error if False and the directory already exists.

    Returns:
        directory (str): The created directory path.
    """

    os.makedirs(directory, exist_ok=exist_ok)
    return directory


def compute_stats(arr: numpy.ndarray, label: str) -> dict[str, float]:
    """Min/max/mean/median/quartiles over finite cells."""
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


def compute_diff(
    starting_data: numpy.ndarray,
    ending_data: numpy.ndarray,
    profile: dict[str, Any],
    output_path: str,
) -> RasterResult:
    diff = ending_data - starting_data
    diff[~numpy.isfinite(diff)] = numpy.nan
    result = save_raster(diff, profile, output_path)
    print(f"  diff saved → {output_path}")
    return result


def valid_mask(arr: NDArray[numpy.floating], nodata: Any) -> NDArray[numpy.bool]:
    """
    Returns true where arr is finite and not nodata.
    """

    if nodata is None:
        return numpy.isfinite(arr)

    return numpy.isfinite(arr) & (arr != nodata)


def safe_mean(arr: NDArray[numpy.floating], nodata: Any) -> float:
    """
    Mean of finite, non-negative, non-nodata values.
    """

    return float(numpy.mean(arr[valid_mask(arr, nodata) & (arr >= 0)]))


def clip_raster(
    data: numpy.ndarray,
    profile: dict[str, Any],
    raster_mask: geopandas.GeoSeries,
    output_path: str,
) -> RasterResult:
    """
    Clips the provided raster using the provided mask.

    Parameters:
      data (numpy.ndarray): The raster data to clip.
      profile (dict): The raster's profile.
      raster_mask (geopandas.GeoSeries): The mask to clip to.
      output_path (str): The path to the output file.

    Returns:
      RasterResult: The clipped data, its updated profile, and output_path.
    """

    with io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data, 1)
            raster_mask = raster_mask.to_crs(dataset.crs)
            # all_touched=False: a cell belongs to the polygon containing its centre,
            # so cells on a shared boundary (e.g. upstream/downstream) land on
            # exactly one side instead of being counted on both.
            clipped, clipped_transform = mask.mask(
                dataset,
                raster_mask.geometry,
                all_touched=False,
                crop=True,
                nodata=numpy.nan,
            )

    clipped_profile = dict(profile)
    clipped_profile.update(height=clipped.shape[1], transform=clipped_transform, width=clipped.shape[2])
    clipped = clipped.squeeze()

    return save_raster(clipped, clipped_profile, output_path)


def reproject_raster(
    data: numpy.ndarray,
    source_profile: dict[str, Any],
    reprojected_crs: str,
    reference_profile: dict[str, Any],
    output_path: str,
    dtype: type = numpy.float32,
    nodata: Any = numpy.nan,
    resampling_method: enums.Resampling = enums.Resampling.mode,
    scale_factor: float = 1.0,
) -> RasterResult:
    """
    Reprojects (and resamples if necessary) the provided raster.

    Parameters:
      data (numpy.ndarray): The raster data to reproject.
      source_profile (dict): The raster's profile.
      reprojected_crs (str): The target CRS for the generated rasters.
      reference_profile (dict): The profile (grid/transform/size) to reproject onto.
      output_path (str): The path to the output file.
      dtype (type): The datatype the raster should be in.
      nodata (Any): The nodata value to use in the raster.
      resampling_method (enums.Resampling): The raster resampling method to use. Defaults to enums.Resampling.average.

    Returns:
      RasterResult: The reprojected data, its updated profile, and output_path.
    """

    profile = dict(source_profile)
    profile.update(compress="lzw", driver="GTiff")

    # override nodata safely
    old_nodata = profile.get("nodata")
    if old_nodata is not None and old_nodata != nodata:
        if numpy.isnan(old_nodata):
            invalid = numpy.isnan(data)
        else:
            invalid = data == old_nodata
        data = numpy.where(invalid, nodata, data)
    profile.update(nodata=nodata)

    data = data.astype(dtype, copy=False)
    profile.update(dtype=dtype)

    dst_transform = reference_profile["transform"]
    width = reference_profile["width"]
    height = reference_profile["height"]

    profile.update(crs=reprojected_crs, height=height, transform=dst_transform, width=width)

    destination = numpy.full((height, width), nodata, dtype=dtype)

    warp.reproject(
        data,
        destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        src_nodata=nodata,
        dst_transform=dst_transform,
        dst_crs=reprojected_crs,
        dst_nodata=nodata,
        resampling=resampling_method,
    )

    if scale_factor != 1.0:
        valid: numpy.ndarray = (
            numpy.isfinite(destination) if (nodata is None or numpy.isnan(nodata)) else (destination != nodata)
        )
        destination = numpy.where(valid, destination * scale_factor, destination).astype(dtype)

    return save_raster(destination, profile, output_path)


def convert_mgd_to_mm_per_year(
    data: numpy.ndarray,
    nodata: Any,
    profile: dict[str, Any],
    year: str,
    output_path: str,
) -> RasterResult:
    cell_area_m2 = abs(profile["transform"].a * profile["transform"].e)

    is_nodata = numpy.isnan(data) if (nodata is None or numpy.isnan(nodata)) else data == nodata
    # 1 MGD × days/yr × 3785.411784 m³/(M gal) × 1000 mm/m / cell_area_m² = mm/yr
    conversion_factor = (366 if int(year) % 4 == 0 else 365) * 3785.411784 * 1000 / cell_area_m2
    result = numpy.where(is_nodata, numpy.nan, data * conversion_factor).astype(numpy.float32)

    return save_raster(result, profile, output_path)


def read_raster(path: str) -> tuple[numpy.ndarray, Any, dict[str, Any]]:
    with rasterio.open(path) as raster:
        data = raster.read(1, masked=True).filled(numpy.nan)
        return data, raster.nodata, raster.profile


def read_raster_profile(path: str) -> dict[str, Any]:
    """Reads only a raster's profile (crs/transform/size/...), without loading pixel data."""
    with rasterio.open(path) as raster:
        return raster.profile


def save_raster(arr: numpy.ndarray, profile: dict[str, Any], path: str) -> RasterResult:
    profile = dict(profile)
    profile.update(dtype=rasterio.float32, nodata=numpy.nan)
    out = numpy.where(numpy.isfinite(arr), arr, numpy.nan).astype(numpy.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)

    return out, profile, path


def sum_rasters(arrays: list[numpy.ndarray], profile: dict[str, Any], output_path: str) -> RasterResult:
    """Sum multiple rasters cell-by-cell. Cells that are nodata in all inputs stay nodata;
    cells with at least one valid value are summed over valid inputs only."""
    for arr in arrays:
        if arr.shape != arrays[0].shape:
            raise ValueError(f"Shape mismatch: expected {arrays[0].shape}, got {arr.shape}")

    stack = numpy.stack(arrays)
    result = numpy.nansum(stack, axis=0).astype(numpy.float64)
    result[numpy.all(numpy.isnan(stack), axis=0)] = numpy.nan

    raster_result = save_raster(result, profile, output_path)
    print(f"Sum raster saved → {output_path}")
    return raster_result


def plot_raster_map(
    data: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: str,
    *,
    title: str,
    cmap: Colormap,
    norm: Normalize,
    cbar_label: str,
    basemap: bool = False,
    cbar_ticks: Sequence[float] | None = None,
    cbar_ticklabels: Sequence[str] | None = None,
) -> None:
    """Render a single-band raster over a HUC boundary map and save it as a PNG.

    Shared scaffold for the RWY / WRC / WRD maps: the caller supplies the
    already-classified/normalized array plus its ``cmap`` and ``norm``; this
    handles bounds, the outside-HUC white mask, optional basemap, HUC outlines,
    equal aspect, and the colorbar.
    """
    cmap.set_bad(alpha=0)  # transparent nodata

    height, width = data.shape
    left, bottom, right, top = transform.array_bounds(height, width, profile["transform"])
    hucs_proj = hucs.to_crs(profile["crs"])
    huc_union = hucs_proj.geometry.union_all()

    # Frame the map on the raster AND the HUC outlines: the raster is often clipped
    # to a subset of the HUCs (e.g. upstream only), and framing on it alone cuts off
    # the rest of the overlay. A small margin keeps edge outlines from being halved.
    huc_left, huc_bottom, huc_right, huc_top = hucs_proj.total_bounds
    view_left, view_bottom = min(left, huc_left), min(bottom, huc_bottom)
    view_right, view_top = max(right, huc_right), max(top, huc_top)
    margin = 0.02 * max(view_right - view_left, view_top - view_bottom)
    view_left, view_bottom = view_left - margin, view_bottom - margin
    view_right, view_top = view_right + margin, view_top + margin

    # Pad large enough to cover any axis expansion from set_aspect("equal"):
    # worst case is (max_dim - min_dim) / 2, so 2× max_dim is always sufficient.
    pad = max(view_right - view_left, view_top - view_bottom) * 2
    outside_mask = shapely_box(view_left - pad, view_bottom - pad, view_right + pad, view_top + pad).difference(
        huc_union
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    def pin_limits() -> None:
        # geopandas/contextily calls can auto-scale the axes to the geometry
        # bounds; re-pin after each one to keep the full view extent.
        ax.set_xlim(view_left, view_right)
        ax.set_ylim(view_bottom, view_top)

    pin_limits()
    if basemap:
        contextily.add_basemap(
            ax,
            crs=profile["crs"],
            source=contextily.providers.Esri.WorldShadedRelief,
            zorder=0,
        )
    else:
        ax.set_facecolor("white")
    pin_limits()

    # aspect=None prevents geopandas from calling ax.set_aspect internally.
    geopandas.GeoSeries([outside_mask], crs=profile["crs"]).plot(ax=ax, color="white", zorder=1, aspect=None)
    pin_limits()

    img = ax.imshow(
        data.astype("float32"),
        extent=[left, right, bottom, top],
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )
    hucs_proj.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=3, aspect=None)
    pin_limits()
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

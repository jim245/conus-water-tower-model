import glob
from os import path

import geopandas
import numpy
from rasterio import enums

import utilities


def prepare_irr_rasters(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
    region_tag: str,
) -> utilities.RasterResult:
    output_directory = utilities.create_directory(
        path.join(root_output_directory, "irr")
    )

    data, nodata, profile = utilities.read_raster(
        path.join("input", "irr", f"irr_huc12_{year}.tif")
    )
    converted, converted_profile, _ = utilities.convert_mgd_to_mm_per_year(
        data,
        nodata,
        profile,
        str(year),
        path.join(output_directory, f"{year}-mm-per-year.tif"),
    )
    return utilities.clip_raster(
        converted,
        converted_profile,
        hucs.geometry,
        path.join(output_directory, f"{year}-{region_tag}.tif"),
    )


def average_quickflow_raster(
    quickflow_model: str,
    quickflow_year: int,
    root_output_directory: str,
) -> utilities.RasterResult:
    """Cell-wise mean of every GCM's quickflow for one scenario (e.g. SSP245) and
    year, on Quickflow's native grid."""

    output_directory = utilities.create_directory(
        path.join(root_output_directory, "quickflow", quickflow_model)
    )

    quickflow_raster_paths = glob.glob(
        path.join(
            "input",
            "Quickflow_Paige",
            f"QF_*_{quickflow_model}",
            f"Qf_{quickflow_year}.tif",
        )
    )

    quickflow_raster_data = list[numpy.ndarray]()
    profile = None

    for filepath in quickflow_raster_paths:
        raster_data, _, profile = utilities.read_raster(filepath)
        quickflow_raster_data.append(raster_data)

    assert profile is not None

    quickflow_stack = numpy.stack(quickflow_raster_data)
    valid_counts = numpy.sum(~numpy.isnan(quickflow_stack), axis=0)
    with numpy.errstate(invalid="ignore"):
        # 0 / 0 for cells nodata in every model quietly yields nan, which is what we want.
        averaged_quickflow = numpy.nansum(quickflow_stack, axis=0) / valid_counts
    return utilities.save_raster(
        averaged_quickflow,
        profile,
        path.join(output_directory, f"{quickflow_model}_{quickflow_year}_averaged.tif"),
    )


def prepare_quickflow_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    quickflow_model: str,
    quickflow_year: int,
    root_output_directory: str,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    """ """

    output_directory = utilities.create_directory(
        path.join(root_output_directory, "quickflow", quickflow_model)
    )

    averaged_quickflow, averaged_profile, _ = average_quickflow_raster(
        quickflow_model, quickflow_year, root_output_directory
    )

    downstream = utilities.clip_raster(
        averaged_quickflow,
        averaged_profile,
        downstream_hucs.geometry,
        path.join(
            output_directory,
            f"{quickflow_model}_{quickflow_year}_averaged_downstream.tif",
        ),
    )
    upstream = utilities.clip_raster(
        averaged_quickflow,
        averaged_profile,
        upstream_hucs.geometry,
        path.join(
            output_directory,
            f"{quickflow_model}_{quickflow_year}_averaged_upstream.tif",
        ),
    )

    return downstream, upstream


def prepare_quickflow_runoff_raster(
    climate_model: str,
    year: int,
    root_output_directory: str,
) -> utilities.RasterResult:
    """The GCM-averaged quickflow for `climate_model`/`year`, used as WRC/WRD's
    runoff: reprojected onto the pws-tot grid so it lines up cell-for-cell with the
    water-use rasters, and converted to mm/yr."""
    runoff_output_directory = utilities.create_directory(
        path.join(root_output_directory, "runoff", climate_model)
    )

    data, profile, _ = average_quickflow_raster(
        climate_model, year, root_output_directory
    )
    reference_profile = utilities.read_raster_profile(
        path.join("input", "pws-tot", f"pws-tot_huc12_{year}.tif")
    )

    return utilities.reproject_raster(
        data,
        profile,
        "EPSG:5070",
        reference_profile,
        path.join(runoff_output_directory, f"{year}-reprojected.tif"),
        resampling_method=enums.Resampling.average,
        scale_factor=1000,  # m/yr to mm/yr, same units as the RO_2013 it replaces
    )


def prepare_runoff_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: str,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    runoff_output_directory = path.join(root_output_directory, "runoff", climate_model)

    reprojected, reprojected_profile, _ = prepare_quickflow_runoff_raster(
        climate_model, year, root_output_directory
    )

    upstream = utilities.clip_raster(
        reprojected,
        reprojected_profile,
        upstream_hucs.geometry,
        path.join(runoff_output_directory, f"{year}-upstream.tif"),
    )
    downstream = utilities.clip_raster(
        reprojected,
        reprojected_profile,
        downstream_hucs.geometry,
        path.join(runoff_output_directory, f"{year}-downstream.tif"),
    )

    return upstream, downstream


def prepare_runoff_raster_basin(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: str,
) -> utilities.RasterResult:
    reprojected, reprojected_profile, _ = prepare_quickflow_runoff_raster(
        climate_model, year, root_output_directory
    )

    return utilities.clip_raster(
        reprojected,
        reprojected_profile,
        hucs.geometry,
        path.join(root_output_directory, "runoff", climate_model, f"{year}-basin.tif"),
    )


def prepare_thermoelectric_rasters(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
    region_tag: str,
    reference_raster_path: str,
) -> utilities.RasterResult:
    thermoelectric_output_dir = utilities.create_directory(
        path.join(root_output_directory, "thermoelectric", str(year))
    )

    data, _, profile = utilities.read_raster(
        path.join(
            "input",
            "thermoelectric",
            f"{year}-annual-consumption.tif",
        )
    )
    clipped, clip_profile, _ = utilities.clip_raster(
        data,
        profile,
        hucs.geometry,
        path.join(
            thermoelectric_output_dir,
            f"{year}-consumption-annual-clipped-{region_tag}.tif",
        ),
    )

    reference_profile = utilities.read_raster_profile(reference_raster_path)

    # Resample while still in MGD (a volume, so summing cells is correct), then
    # convert to mm/yr using the destination grid's cell area.
    reprojected, reprojected_profile, _ = utilities.reproject_raster(
        clipped,
        clip_profile,
        "EPSG:5070",
        reference_profile,
        path.join(
            thermoelectric_output_dir,
            f"{year}-consumption-annual-reprojected-{region_tag}.tif",
        ),
        resampling_method=enums.Resampling.sum,
    )

    return utilities.convert_mgd_to_mm_per_year(
        reprojected,
        reprojected_profile["nodata"],
        reprojected_profile,
        str(year),
        path.join(
            thermoelectric_output_dir,
            f"{year}-consumption-annual-mm-per-year-{region_tag}.tif",
        ),
    )


def prepare_pws_withdrawal_raster(
    root_output_directory: str,
    year: int,
) -> utilities.RasterResult:
    """pws-sw (surface-water public-supply withdrawal) in mm/yr, on the full grid.

    Surface water only, matching the notebook's SW_WD input: groundwater
    withdrawals don't come out of the runoff these indices compare against. Not
    withdrawal minus pws-cu: that difference is return flow (water discharged back
    downstream), not water taken out of the river."""
    pws_wd_output_directory = utilities.create_directory(
        path.join(root_output_directory, "pws-wd")
    )

    # path: input/pws-sw/pws-sw_huc12_{year}.tif
    sw_data, sw_nodata, sw_profile = utilities.read_raster(
        path.join("input", "pws-sw", f"pws-sw_huc12_{year}.tif")
    )
    return utilities.convert_mgd_to_mm_per_year(
        sw_data,
        sw_nodata,
        sw_profile,
        str(year),
        path.join(pws_wd_output_directory, f"{year}.tif"),
    )


def prepare_pws_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int = 2010,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    pws_output_directory = utilities.create_directory(
        path.join(root_output_directory, "pws")
    )

    # path: input/pws-cu/pws-cu_huc12_{year}.tif
    cu_data, cu_nodata, cu_profile = utilities.read_raster(
        path.join("input", "pws-cu", f"pws-cu_huc12_{year}.tif")
    )
    converted_cu, converted_cu_profile, _ = utilities.convert_mgd_to_mm_per_year(
        cu_data,
        cu_nodata,
        cu_profile,
        str(year),
        path.join(pws_output_directory, f"{year}-cu-mm-per-year.tif"),
    )

    # pws-cu
    pws_cu_result = utilities.clip_raster(
        converted_cu,
        converted_cu_profile,
        upstream_hucs.geometry,
        path.join(pws_output_directory, f"{year}-cu-upstream.tif"),
    )

    # pws-wd
    pws_wd_data, pws_wd_profile, _ = prepare_pws_withdrawal_raster(
        root_output_directory, year
    )
    pws_wd_result = utilities.clip_raster(
        pws_wd_data,
        pws_wd_profile,
        downstream_hucs.geometry,
        path.join(root_output_directory, "pws-wd", f"{year}-downstream.tif"),
    )

    return pws_cu_result, pws_wd_result


def prepare_withdrawal_raster_basin(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
) -> utilities.RasterResult:
    """pws-sw (surface-water public-supply withdrawal), basin-wide."""
    pws_wd, pws_wd_profile, _ = prepare_pws_withdrawal_raster(
        root_output_directory, year
    )

    return utilities.clip_raster(
        pws_wd,
        pws_wd_profile,
        hucs.geometry,
        path.join(root_output_directory, "pws-wd", f"{year}-basin.tif"),
    )


def prepare_wrd_consumption_raster(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
) -> utilities.RasterResult:
    """irr + pws-cu consumptive-use + thermoelectric consumption, basin-wide."""
    irr_raster, irr_profile, irr_path = prepare_irr_rasters(
        hucs, root_output_directory, year, region_tag="basin"
    )

    consumption_output_directory = utilities.create_directory(
        path.join(root_output_directory, "pws")
    )
    cu_data, cu_nodata, cu_profile = utilities.read_raster(
        path.join("input", "pws-cu", f"pws-cu_huc12_{year}.tif")
    )
    converted_cu, converted_cu_profile, _ = utilities.convert_mgd_to_mm_per_year(
        cu_data,
        cu_nodata,
        cu_profile,
        str(year),
        path.join(consumption_output_directory, f"{year}-cu-mm-per-year.tif"),
    )
    pws_cu_basin, _, _ = utilities.clip_raster(
        converted_cu,
        converted_cu_profile,
        hucs.geometry,
        path.join(consumption_output_directory, f"{year}-cu-basin.tif"),
    )

    thermo_basin, _, _ = prepare_thermoelectric_rasters(
        hucs,
        root_output_directory,
        year,
        region_tag="basin",
        reference_raster_path=irr_path,
    )

    return utilities.sum_rasters(
        [irr_raster, pws_cu_basin, thermo_basin],
        irr_profile,
        path.join(root_output_directory, f"{year}-wrd-consumption-basin.tif"),
    )

"""
Prepares the input rasters of RWY, WRC and WRD for one basin: runoff from Quickflow, and water consumption and
withdrawal from the water-use inputs.

Each function reads its inputs from input/ and writes each step under root_output_directory. Water-use inputs are annual
means in MGD, and the functions convert them to mm/yr. Quickflow has no unit metadata: the functions assume m/yr.
"""

from pathlib import Path

import geopandas
import numpy
from rasterio import enums

import utilities

QUICKFLOW_DIRECTORY = Path("input", "Quickflow_Paige")


def prepare_irr_rasters(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
    region_tag: str,
) -> utilities.RasterResult:
    """
    Converts the irrigation consumption to mm/yr and clips it to the HUCs.

    Writes irr/<year>-mm-per-year.tif (full grid) and irr/<year>-<region_tag>.tif under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The HUCs to clip to.
        root_output_directory (str): The output directory of the basin.
        year (int): The year of the data.
        region_tag (str): The name of the clip region in the file name, for example upstream or basin.

    Returns:
        result (RasterResult): The clipped irrigation consumption, in mm/yr.
    """

    output_directory = utilities.create_directory(str(Path(root_output_directory, "irr")))

    data, nodata, profile = utilities.read_raster(str(Path("input", "irr", f"irr_huc12_{year}.tif")))
    converted, converted_profile, _ = utilities.convert_mgd_to_mm_per_year(
        data,
        nodata,
        profile,
        year,
        str(Path(output_directory, f"{year}-mm-per-year.tif")),
    )
    return utilities.clip_raster(
        converted,
        converted_profile,
        hucs.geometry,
        str(Path(output_directory, f"{year}-{region_tag}.tif")),
    )


def average_quickflow_raster(
    quickflow_model: str,
    quickflow_year: int,
    root_output_directory: str,
) -> utilities.RasterResult:
    """
    Computes the cell-by-cell mean Quickflow over the GCMs of one scenario, on the Quickflow grid.

    A cell gets the mean of the GCMs that have a value there. A cell with no value in any GCM is NaN. Writes
    quickflow/<quickflow_model>/<quickflow_model>_<quickflow_year>_averaged.tif under root_output_directory.

    Parameters:
        quickflow_model (str): The scenario, for example SSP245. Each input/Quickflow_Paige/QF_<GCM>_<quickflow_model>
            folder is one GCM.
        quickflow_year (int): The year of the data.
        root_output_directory (str): The output directory of the basin.

    Returns:
        result (RasterResult): The mean Quickflow, in m/yr.

    Raises:
        FileNotFoundError: If no GCM folder exists for the scenario, or a GCM folder does not have exactly one file for
            the year.
    """

    output_directory = utilities.create_directory(str(Path(root_output_directory, "quickflow", quickflow_model)))

    gcm_directories = sorted(QUICKFLOW_DIRECTORY.glob(f"QF_*_{quickflow_model}"))
    if not gcm_directories:
        raise FileNotFoundError(f"no QF_*_{quickflow_model} folder in {QUICKFLOW_DIRECTORY}")

    quickflow_paths = list[Path]()
    for gcm_directory in gcm_directories:
        # the file name starts with Qf_ or QF_, by GCM
        year_paths = list(gcm_directory.glob(f"Qf_{quickflow_year}.tif", case_sensitive=False))
        if len(year_paths) != 1:
            raise FileNotFoundError(
                f"{gcm_directory} has {len(year_paths)} Quickflow files for {quickflow_year}, not 1"
            )
        quickflow_paths.extend(year_paths)

    rasters = [utilities.read_raster(str(quickflow_path)) for quickflow_path in quickflow_paths]
    _, _, profile = rasters[0]

    quickflow_stack = numpy.stack([data for data, _, _ in rasters])
    valid_counts = numpy.sum(~numpy.isnan(quickflow_stack), axis=0)
    # a cell with no value in any GCM gets 0 / 0, thus NaN
    with numpy.errstate(invalid="ignore"):
        averaged_quickflow = numpy.nansum(quickflow_stack, axis=0) / valid_counts
    return utilities.save_raster(
        averaged_quickflow,
        profile,
        str(Path(output_directory, f"{quickflow_model}_{quickflow_year}_averaged.tif")),
    )


def prepare_quickflow_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    quickflow_model: str,
    quickflow_year: int,
    root_output_directory: str,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    """
    Clips the mean Quickflow (average_quickflow_raster) to the upstream HUCs and to the downstream HUCs, on the
    Quickflow grid.

    Writes <quickflow_model>_<quickflow_year>_averaged_upstream.tif and _downstream.tif in
    quickflow/<quickflow_model>/ under root_output_directory.

    Parameters:
        upstream_hucs (geopandas.GeoDataFrame): The upstream HUCs.
        downstream_hucs (geopandas.GeoDataFrame): The downstream HUCs.
        quickflow_model (str): The scenario, for example SSP245.
        quickflow_year (int): The year of the data.
        root_output_directory (str): The output directory of the basin.

    Returns:
        upstream (RasterResult): The upstream Quickflow, in m/yr.
        downstream (RasterResult): The downstream Quickflow, in m/yr.
    """

    output_directory = utilities.create_directory(str(Path(root_output_directory, "quickflow", quickflow_model)))

    averaged_quickflow, averaged_profile, _ = average_quickflow_raster(
        quickflow_model, quickflow_year, root_output_directory
    )

    upstream = utilities.clip_raster(
        averaged_quickflow,
        averaged_profile,
        upstream_hucs.geometry,
        str(Path(output_directory, f"{quickflow_model}_{quickflow_year}_averaged_upstream.tif")),
    )
    downstream = utilities.clip_raster(
        averaged_quickflow,
        averaged_profile,
        downstream_hucs.geometry,
        str(Path(output_directory, f"{quickflow_model}_{quickflow_year}_averaged_downstream.tif")),
    )

    return upstream, downstream


def prepare_quickflow_runoff_raster(
    climate_model: str,
    year: int,
    root_output_directory: str,
) -> utilities.RasterResult:
    """
    Reprojects the mean Quickflow (average_quickflow_raster) onto the pws-tot grid, in mm/yr, for use as the runoff of
    WRC and WRD.

    On the pws-tot grid, the runoff has the same cells as the water-use rasters. Writes
    runoff/<climate_model>/<year>-reprojected.tif under root_output_directory.

    Parameters:
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.
        root_output_directory (str): The output directory of the basin.

    Returns:
        result (RasterResult): The runoff on the full pws-tot grid, in mm/yr.
    """

    runoff_output_directory = utilities.create_directory(str(Path(root_output_directory, "runoff", climate_model)))

    data, profile, _ = average_quickflow_raster(climate_model, year, root_output_directory)
    reference_profile = utilities.read_raster_profile(str(Path("input", "pws-tot", f"pws-tot_huc12_{year}.tif")))

    return utilities.reproject_raster(
        data,
        profile,
        reference_profile,
        str(Path(runoff_output_directory, f"{year}-reprojected.tif")),
        resampling_method=enums.Resampling.average,
        scale_factor=utilities.MM_PER_M,
    )


def prepare_runoff_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: str,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    """
    Clips the runoff (prepare_quickflow_runoff_raster) to the upstream HUCs and to the downstream HUCs.

    Writes runoff/<climate_model>/<year>-upstream.tif and <year>-downstream.tif under root_output_directory.

    Parameters:
        upstream_hucs (geopandas.GeoDataFrame): The upstream HUCs.
        downstream_hucs (geopandas.GeoDataFrame): The downstream HUCs.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.
        root_output_directory (str): The output directory of the basin.

    Returns:
        upstream (RasterResult): The upstream runoff, in mm/yr.
        downstream (RasterResult): The downstream runoff, in mm/yr.
    """

    runoff_output_directory = Path(root_output_directory, "runoff", climate_model)

    reprojected, reprojected_profile, _ = prepare_quickflow_runoff_raster(climate_model, year, root_output_directory)

    upstream = utilities.clip_raster(
        reprojected,
        reprojected_profile,
        upstream_hucs.geometry,
        str(runoff_output_directory / f"{year}-upstream.tif"),
    )
    downstream = utilities.clip_raster(
        reprojected,
        reprojected_profile,
        downstream_hucs.geometry,
        str(runoff_output_directory / f"{year}-downstream.tif"),
    )

    return upstream, downstream


def prepare_runoff_raster_basin(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: str,
) -> utilities.RasterResult:
    """
    Clips the runoff (prepare_quickflow_runoff_raster) to the basin HUCs.

    Writes runoff/<climate_model>/<year>-basin.tif under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The basin HUCs.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.
        root_output_directory (str): The output directory of the basin.

    Returns:
        result (RasterResult): The basin runoff, in mm/yr.
    """

    reprojected, reprojected_profile, _ = prepare_quickflow_runoff_raster(climate_model, year, root_output_directory)

    return utilities.clip_raster(
        reprojected,
        reprojected_profile,
        hucs.geometry,
        str(Path(root_output_directory, "runoff", climate_model, f"{year}-basin.tif")),
    )


def prepare_thermoelectric_rasters(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
    region_tag: str,
    reference_raster_path: str,
) -> utilities.RasterResult:
    """
    Clips the thermoelectric consumption to the HUCs, moves it onto the reference grid, and converts it to mm/yr.

    Writes <year>-consumption-annual-clipped-<region_tag>.tif, -reprojected-<region_tag>.tif and
    -mm-per-year-<region_tag>.tif in thermoelectric/<year>/ under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The HUCs to clip to.
        root_output_directory (str): The output directory of the basin.
        year (int): The year of the data.
        region_tag (str): The name of the clip region in the file name, for example upstream or basin.
        reference_raster_path (str): The raster whose grid the output gets.

    Returns:
        result (RasterResult): The thermoelectric consumption on the reference grid, in mm/yr.
    """

    thermoelectric_output_dir = utilities.create_directory(
        str(Path(root_output_directory, "thermoelectric", str(year)))
    )

    data, _, profile = utilities.read_raster(str(Path("input", "thermoelectric", f"{year}-annual-consumption.tif")))
    clipped, clip_profile, _ = utilities.clip_raster(
        data,
        profile,
        hucs.geometry,
        str(Path(thermoelectric_output_dir, f"{year}-consumption-annual-clipped-{region_tag}.tif")),
    )

    reference_profile = utilities.read_raster_profile(reference_raster_path)

    # resample in MGD, a volume for each cell, so that sum keeps the total. The conversion then uses the cell area of
    # the reference grid
    reprojected, reprojected_profile, _ = utilities.reproject_raster(
        clipped,
        clip_profile,
        reference_profile,
        str(Path(thermoelectric_output_dir, f"{year}-consumption-annual-reprojected-{region_tag}.tif")),
        resampling_method=enums.Resampling.sum,
    )

    return utilities.convert_mgd_to_mm_per_year(
        reprojected,
        numpy.nan,
        reprojected_profile,
        year,
        str(Path(thermoelectric_output_dir, f"{year}-consumption-annual-mm-per-year-{region_tag}.tif")),
    )


def prepare_pws_consumption_raster(
    root_output_directory: str,
    year: int,
) -> utilities.RasterResult:
    """
    Converts the public-supply consumption (pws-cu) to mm/yr, on the full grid.

    Writes pws/<year>-cu-mm-per-year.tif under root_output_directory.

    Parameters:
        root_output_directory (str): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        result (RasterResult): The public-supply consumption, in mm/yr.
    """

    pws_output_directory = utilities.create_directory(str(Path(root_output_directory, "pws")))

    cu_data, cu_nodata, cu_profile = utilities.read_raster(str(Path("input", "pws-cu", f"pws-cu_huc12_{year}.tif")))
    return utilities.convert_mgd_to_mm_per_year(
        cu_data,
        cu_nodata,
        cu_profile,
        year,
        str(Path(pws_output_directory, f"{year}-cu-mm-per-year.tif")),
    )


def prepare_pws_withdrawal_raster(
    root_output_directory: str,
    year: int,
) -> utilities.RasterResult:
    """
    Converts the surface-water public-supply withdrawal (pws-sw) to mm/yr, on the full grid.

    The withdrawal is surface water only, as SW_WD in the notebook: groundwater withdrawals do not come from the runoff
    that the indices compare against. It is not the withdrawal minus pws-cu, because that difference is return flow.

    Writes pws-wd/<year>.tif under root_output_directory.

    Parameters:
        root_output_directory (str): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        result (RasterResult): The surface-water public-supply withdrawal, in mm/yr.
    """

    pws_wd_output_directory = utilities.create_directory(str(Path(root_output_directory, "pws-wd")))

    sw_data, sw_nodata, sw_profile = utilities.read_raster(str(Path("input", "pws-sw", f"pws-sw_huc12_{year}.tif")))
    return utilities.convert_mgd_to_mm_per_year(
        sw_data,
        sw_nodata,
        sw_profile,
        year,
        str(Path(pws_wd_output_directory, f"{year}.tif")),
    )


def prepare_pws_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    """
    Clips the public-supply consumption to the upstream HUCs, and the surface-water public-supply withdrawal to the
    downstream HUCs.

    Writes pws/<year>-cu-upstream.tif and pws-wd/<year>-downstream.tif under root_output_directory.

    Parameters:
        upstream_hucs (geopandas.GeoDataFrame): The upstream HUCs.
        downstream_hucs (geopandas.GeoDataFrame): The downstream HUCs.
        root_output_directory (str): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        upstream_consumption (RasterResult): The upstream public-supply consumption, in mm/yr.
        downstream_withdrawal (RasterResult): The downstream surface-water public-supply withdrawal, in mm/yr.
    """

    cu_data, cu_profile, _ = prepare_pws_consumption_raster(root_output_directory, year)
    upstream_consumption = utilities.clip_raster(
        cu_data,
        cu_profile,
        upstream_hucs.geometry,
        str(Path(root_output_directory, "pws", f"{year}-cu-upstream.tif")),
    )

    wd_data, wd_profile, _ = prepare_pws_withdrawal_raster(root_output_directory, year)
    downstream_withdrawal = utilities.clip_raster(
        wd_data,
        wd_profile,
        downstream_hucs.geometry,
        str(Path(root_output_directory, "pws-wd", f"{year}-downstream.tif")),
    )

    return upstream_consumption, downstream_withdrawal


def prepare_withdrawal_raster_basin(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
) -> utilities.RasterResult:
    """
    Clips the surface-water public-supply withdrawal (prepare_pws_withdrawal_raster) to the basin HUCs.

    Writes pws-wd/<year>-basin.tif under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The basin HUCs.
        root_output_directory (str): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        result (RasterResult): The basin withdrawal, in mm/yr.
    """

    pws_wd, pws_wd_profile, _ = prepare_pws_withdrawal_raster(root_output_directory, year)

    return utilities.clip_raster(
        pws_wd,
        pws_wd_profile,
        hucs.geometry,
        str(Path(root_output_directory, "pws-wd", f"{year}-basin.tif")),
    )


def prepare_wrd_consumption_raster(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: str,
    year: int,
) -> utilities.RasterResult:
    """
    Sums the surface-water consumption (irrigation, public supply and thermoelectric) in the basin HUCs, in mm/yr.

    Exception: the pws-cu input is the public-supply consumption from all sources, thus it includes the consumption
    from groundwater.

    Writes <year>-wrd-consumption-basin.tif under root_output_directory, and the steps in irr/, pws/ and
    thermoelectric/<year>/.

    Parameters:
        hucs (geopandas.GeoDataFrame): The basin HUCs.
        root_output_directory (str): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        result (RasterResult): The basin consumption, on the irrigation grid, in mm/yr.
    """

    irr = prepare_irr_rasters(hucs, root_output_directory, year, region_tag="basin")

    cu_data, cu_profile, _ = prepare_pws_consumption_raster(root_output_directory, year)
    pws_cu_basin = utilities.clip_raster(
        cu_data,
        cu_profile,
        hucs.geometry,
        str(Path(root_output_directory, "pws", f"{year}-cu-basin.tif")),
    )

    _, _, irr_path = irr
    thermo_basin = prepare_thermoelectric_rasters(
        hucs,
        root_output_directory,
        year,
        region_tag="basin",
        reference_raster_path=irr_path,
    )

    return utilities.sum_rasters(
        [irr, pws_cu_basin, thermo_basin],
        str(Path(root_output_directory, f"{year}-wrd-consumption-basin.tif")),
    )

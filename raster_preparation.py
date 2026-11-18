"""
Prepares the input rasters of RWY, WRC and WRD: runoff from Quickflow, and water consumption and withdrawal from the
water-use inputs.

prepare_national_rasters computes the national rasters one time for each run, under NATIONAL_DIRECTORY. These are the
mean Quickflow, the runoff on the water-use grid and the water use in mm/yr. The other functions clip these rasters to the
HUCs of one basin and write each step under root_output_directory. Exception: prepare_thermoelectric_rasters clips the
thermoelectric input itself, and converts it to mm/yr after the clip. Water-use inputs are annual
means in MGD, and the functions convert them to mm/yr. Quickflow has no unit metadata: the functions assume m/yr.
"""

from collections.abc import Sequence
from pathlib import Path

import geopandas
import numpy
from rasterio import enums

import utilities

QUICKFLOW_DIRECTORY = Path("input", "Quickflow_Paige")
NATIONAL_DIRECTORY = Path("output", "national")


def water_use_grid_path(year: int) -> Path:
    """
    Gives the path of the pws-tot input, whose grid is the 1 km grid of the water use, WRC and WRD.

    Parameters:
        year (int): The year of the data.

    Returns:
        path (Path): The path under input/pws-tot/.
    """

    return Path("input", "pws-tot", f"pws-tot_huc12_{year}.tif")


def national_quickflow_path(climate_model: str, year: int) -> Path:
    """
    Gives the path of the mean Quickflow (average_quickflow_raster).

    Parameters:
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.

    Returns:
        path (Path): The path under NATIONAL_DIRECTORY.
    """

    return Path(NATIONAL_DIRECTORY, "quickflow", climate_model, f"{climate_model}_{year}_averaged.tif")


def national_runoff_path(climate_model: str, year: int) -> Path:
    """
    Gives the path of the runoff on the pws-tot grid (prepare_quickflow_runoff_raster).

    Parameters:
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.

    Returns:
        path (Path): The path under NATIONAL_DIRECTORY.
    """

    return Path(NATIONAL_DIRECTORY, "runoff", climate_model, f"{year}-reprojected.tif")


def national_irr_path(year: int) -> Path:
    """
    Gives the path of the irrigation consumption in mm/yr (prepare_irr_raster).

    Parameters:
        year (int): The year of the data.

    Returns:
        path (Path): The path under NATIONAL_DIRECTORY.
    """

    return Path(NATIONAL_DIRECTORY, "irr", f"{year}-mm-per-year.tif")


def national_pws_consumption_path(year: int) -> Path:
    """
    Gives the path of the public-supply consumption in mm/yr (prepare_pws_consumption_raster).

    Parameters:
        year (int): The year of the data.

    Returns:
        path (Path): The path under NATIONAL_DIRECTORY.
    """

    return Path(NATIONAL_DIRECTORY, "pws", f"{year}-cu-mm-per-year.tif")


def national_pws_withdrawal_path(year: int) -> Path:
    """
    Gives the path of the surface-water public-supply withdrawal in mm/yr (prepare_pws_withdrawal_raster).

    Parameters:
        year (int): The year of the data.

    Returns:
        path (Path): The path under NATIONAL_DIRECTORY.
    """

    return Path(NATIONAL_DIRECTORY, "pws-wd", f"{year}.tif")


def read_national_raster(raster_path: Path) -> utilities.RasterResult:
    """
    Reads a raster that prepare_national_rasters wrote.

    Parameters:
        raster_path (Path): The path of the raster (a national_*_path function).

    Returns:
        result (RasterResult): The raster, NaN for no data.

    Raises:
        FileNotFoundError: If the raster does not exist.
    """

    if not raster_path.exists():
        raise FileNotFoundError(f"{raster_path} does not exist: run prepare_national_rasters first")

    data, _, profile = utilities.read_raster(raster_path)
    return data, profile, raster_path


def average_quickflow_raster(quickflow_model: str, quickflow_year: int) -> utilities.RasterResult:
    """
    Computes the cell-by-cell mean Quickflow over the GCMs of one scenario, on the Quickflow grid.

    A cell gets the mean of the GCMs that have a value there. A cell with no value in any GCM is NaN. Writes
    national_quickflow_path.

    Parameters:
        quickflow_model (str): The scenario, for example SSP245. Each input/Quickflow_Paige/QF_<GCM>_<quickflow_model>
            folder is one GCM.
        quickflow_year (int): The year of the data.

    Returns:
        result (RasterResult): The mean Quickflow, in m/yr.

    Raises:
        FileNotFoundError: If no GCM folder exists for the scenario, or a GCM folder does not have exactly one file for
            the year.
    """

    output_path = national_quickflow_path(quickflow_model, quickflow_year)
    utilities.create_directory(output_path.parent)

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

    rasters = [utilities.read_raster(quickflow_path) for quickflow_path in quickflow_paths]
    _, _, profile = rasters[0]

    quickflow_stack = numpy.stack([data for data, _, _ in rasters])
    valid_counts = numpy.sum(~numpy.isnan(quickflow_stack), axis=0)
    # a cell with no value in any GCM gets 0 / 0, thus NaN
    with numpy.errstate(invalid="ignore"):
        averaged_quickflow = numpy.nansum(quickflow_stack, axis=0) / valid_counts
    return utilities.save_raster(averaged_quickflow, profile, output_path)


def prepare_quickflow_runoff_raster(
    climate_model: str,
    year: int,
    averaged: utilities.RasterResult,
) -> utilities.RasterResult:
    """
    Reprojects the mean Quickflow onto the pws-tot grid, in mm/yr, for use as the runoff of WRC and WRD.

    On the pws-tot grid, the runoff has the same cells as the water-use rasters. Writes national_runoff_path.

    Parameters:
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.
        averaged (RasterResult): The mean Quickflow of the scenario and year (average_quickflow_raster).

    Returns:
        result (RasterResult): The runoff on the full pws-tot grid, in mm/yr.
    """

    output_path = national_runoff_path(climate_model, year)
    utilities.create_directory(output_path.parent)

    data, profile, _ = averaged
    reference_profile = utilities.read_raster_profile(water_use_grid_path(year))

    return utilities.reproject_raster(
        data,
        profile,
        reference_profile,
        output_path,
        resampling_method=enums.Resampling.average,
        scale_factor=utilities.MM_PER_M,
    )


def prepare_irr_raster(year: int) -> utilities.RasterResult:
    """
    Converts the irrigation consumption to mm/yr, on the full grid.

    Writes national_irr_path.

    Parameters:
        year (int): The year of the data.

    Returns:
        result (RasterResult): The irrigation consumption, in mm/yr.
    """

    output_path = national_irr_path(year)
    utilities.create_directory(output_path.parent)

    data, nodata, profile = utilities.read_raster(Path("input", "irr", f"irr_huc12_{year}.tif"))
    return utilities.convert_mgd_to_mm_per_year(data, nodata, profile, year, output_path)


def prepare_pws_consumption_raster(year: int) -> utilities.RasterResult:
    """
    Converts the public-supply consumption (pws-cu) to mm/yr, on the full grid.

    Writes national_pws_consumption_path.

    Parameters:
        year (int): The year of the data.

    Returns:
        result (RasterResult): The public-supply consumption, in mm/yr.
    """

    output_path = national_pws_consumption_path(year)
    utilities.create_directory(output_path.parent)

    cu_data, cu_nodata, cu_profile = utilities.read_raster(Path("input", "pws-cu", f"pws-cu_huc12_{year}.tif"))
    return utilities.convert_mgd_to_mm_per_year(cu_data, cu_nodata, cu_profile, year, output_path)


def prepare_pws_withdrawal_raster(year: int) -> utilities.RasterResult:
    """
    Converts the surface-water public-supply withdrawal (pws-sw) to mm/yr, on the full grid.

    The withdrawal is surface water only, as SW_WD in the notebook: groundwater withdrawals do not come from the runoff
    that the indices compare against. It is not the withdrawal minus pws-cu, because that difference is return flow.

    pws-sw has no value (NaN) outside the water service areas. There is no public supply there, thus WRC and WRD count
    such a cell as 0 where the runoff has a value.

    Writes national_pws_withdrawal_path.

    Parameters:
        year (int): The year of the data.

    Returns:
        result (RasterResult): The surface-water public-supply withdrawal, in mm/yr.
    """

    output_path = national_pws_withdrawal_path(year)
    utilities.create_directory(output_path.parent)

    sw_data, sw_nodata, sw_profile = utilities.read_raster(Path("input", "pws-sw", f"pws-sw_huc12_{year}.tif"))
    return utilities.convert_mgd_to_mm_per_year(sw_data, sw_nodata, sw_profile, year, output_path)


def prepare_national_rasters(climate_models: Sequence[str], years: Sequence[int]) -> None:
    """
    Computes the national rasters that the basin functions clip, one time for each scenario and year.

    The water use does not depend on the scenario. Writes the rasters of the national_*_path functions, and does not
    keep them in memory.

    Parameters:
        climate_models (Sequence[str]): The scenarios, for example SSP245.
        years (Sequence[int]): The years.
    """

    for year in years:
        prepare_irr_raster(year)
        prepare_pws_consumption_raster(year)
        prepare_pws_withdrawal_raster(year)
        for climate_model in climate_models:
            averaged = average_quickflow_raster(climate_model, year)
            prepare_quickflow_runoff_raster(climate_model, year, averaged)


def prepare_irr_rasters(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: Path,
    year: int,
    region_tag: str,
) -> utilities.RasterResult:
    """
    Clips the irrigation consumption in mm/yr (prepare_irr_raster) to the HUCs.

    Writes irr/<year>-<region_tag>.tif under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The HUCs to clip to.
        root_output_directory (Path): The output directory of the basin.
        year (int): The year of the data.
        region_tag (str): The name of the clip region in the file name, for example upstream or basin.

    Returns:
        result (RasterResult): The clipped irrigation consumption, in mm/yr.
    """

    output_directory = utilities.create_directory(Path(root_output_directory, "irr"))

    converted, converted_profile, _ = read_national_raster(national_irr_path(year))
    return utilities.clip_raster(
        converted,
        converted_profile,
        hucs.geometry,
        Path(output_directory, f"{year}-{region_tag}.tif"),
    )


def prepare_quickflow_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    quickflow_model: str,
    quickflow_year: int,
    root_output_directory: Path,
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
        root_output_directory (Path): The output directory of the basin.

    Returns:
        upstream (RasterResult): The upstream Quickflow, in m/yr.
        downstream (RasterResult): The downstream Quickflow, in m/yr.
    """

    output_directory = utilities.create_directory(Path(root_output_directory, "quickflow", quickflow_model))

    averaged_quickflow, averaged_profile, _ = read_national_raster(
        national_quickflow_path(quickflow_model, quickflow_year)
    )

    upstream = utilities.clip_raster(
        averaged_quickflow,
        averaged_profile,
        upstream_hucs.geometry,
        Path(output_directory, f"{quickflow_model}_{quickflow_year}_averaged_upstream.tif"),
    )
    downstream = utilities.clip_raster(
        averaged_quickflow,
        averaged_profile,
        downstream_hucs.geometry,
        Path(output_directory, f"{quickflow_model}_{quickflow_year}_averaged_downstream.tif"),
    )

    return upstream, downstream


def prepare_runoff_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: Path,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    """
    Clips the runoff (prepare_quickflow_runoff_raster) to the upstream HUCs and to the downstream HUCs.

    Writes runoff/<climate_model>/<year>-upstream.tif and <year>-downstream.tif under root_output_directory.

    Parameters:
        upstream_hucs (geopandas.GeoDataFrame): The upstream HUCs.
        downstream_hucs (geopandas.GeoDataFrame): The downstream HUCs.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.
        root_output_directory (Path): The output directory of the basin.

    Returns:
        upstream (RasterResult): The upstream runoff, in mm/yr.
        downstream (RasterResult): The downstream runoff, in mm/yr.
    """

    runoff_output_directory = utilities.create_directory(Path(root_output_directory, "runoff", climate_model))

    reprojected, reprojected_profile, _ = read_national_raster(national_runoff_path(climate_model, year))

    upstream = utilities.clip_raster(
        reprojected,
        reprojected_profile,
        upstream_hucs.geometry,
        runoff_output_directory / f"{year}-upstream.tif",
    )
    downstream = utilities.clip_raster(
        reprojected,
        reprojected_profile,
        downstream_hucs.geometry,
        runoff_output_directory / f"{year}-downstream.tif",
    )

    return upstream, downstream


def prepare_runoff_raster_basin(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: Path,
) -> utilities.RasterResult:
    """
    Clips the runoff (prepare_quickflow_runoff_raster) to the basin HUCs.

    Writes runoff/<climate_model>/<year>-basin.tif under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The basin HUCs.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year of the data.
        root_output_directory (Path): The output directory of the basin.

    Returns:
        result (RasterResult): The basin runoff, in mm/yr.
    """

    runoff_output_directory = utilities.create_directory(Path(root_output_directory, "runoff", climate_model))

    reprojected, reprojected_profile, _ = read_national_raster(national_runoff_path(climate_model, year))

    return utilities.clip_raster(
        reprojected,
        reprojected_profile,
        hucs.geometry,
        Path(runoff_output_directory, f"{year}-basin.tif"),
    )


def prepare_thermoelectric_rasters(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: Path,
    year: int,
    region_tag: str,
    reference_raster_path: Path,
) -> utilities.RasterResult:
    """
    Clips the thermoelectric consumption to the HUCs, moves it onto the reference grid, and converts it to mm/yr.

    Writes <year>-consumption-annual-clipped-<region_tag>.tif, -reprojected-<region_tag>.tif and
    -mm-per-year-<region_tag>.tif in thermoelectric/<year>/ under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The HUCs to clip to.
        root_output_directory (Path): The output directory of the basin.
        year (int): The year of the data.
        region_tag (str): The name of the clip region in the file name, for example upstream or basin.
        reference_raster_path (Path): The raster whose grid the output gets.

    Returns:
        result (RasterResult): The thermoelectric consumption on the reference grid, in mm/yr.
    """

    thermoelectric_output_dir = utilities.create_directory(Path(root_output_directory, "thermoelectric", str(year)))

    data, _, profile = utilities.read_raster(Path("input", "thermoelectric", f"{year}-annual-consumption.tif"))
    clipped, clip_profile, _ = utilities.clip_raster(
        data,
        profile,
        hucs.geometry,
        Path(thermoelectric_output_dir, f"{year}-consumption-annual-clipped-{region_tag}.tif"),
    )

    reference_profile = utilities.read_raster_profile(reference_raster_path)

    # resample in MGD, a volume for each cell, so that sum keeps the total. The conversion then uses the cell area of
    # the reference grid
    reprojected, reprojected_profile, _ = utilities.reproject_raster(
        clipped,
        clip_profile,
        reference_profile,
        Path(thermoelectric_output_dir, f"{year}-consumption-annual-reprojected-{region_tag}.tif"),
        resampling_method=enums.Resampling.sum,
    )

    return utilities.convert_mgd_to_mm_per_year(
        reprojected,
        numpy.nan,
        reprojected_profile,
        year,
        Path(thermoelectric_output_dir, f"{year}-consumption-annual-mm-per-year-{region_tag}.tif"),
    )


def prepare_pws_rasters(
    upstream_hucs: geopandas.GeoDataFrame,
    downstream_hucs: geopandas.GeoDataFrame,
    root_output_directory: Path,
    year: int,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    """
    Clips the public-supply consumption to the upstream HUCs, and the surface-water public-supply withdrawal to the
    downstream HUCs.

    Writes pws/<year>-cu-upstream.tif and pws-wd/<year>-downstream.tif under root_output_directory.

    Parameters:
        upstream_hucs (geopandas.GeoDataFrame): The upstream HUCs.
        downstream_hucs (geopandas.GeoDataFrame): The downstream HUCs.
        root_output_directory (Path): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        upstream_consumption (RasterResult): The upstream public-supply consumption, in mm/yr.
        downstream_withdrawal (RasterResult): The downstream surface-water public-supply withdrawal, in mm/yr.
    """

    pws_output_directory = utilities.create_directory(Path(root_output_directory, "pws"))
    pws_wd_output_directory = utilities.create_directory(Path(root_output_directory, "pws-wd"))

    cu_data, cu_profile, _ = read_national_raster(national_pws_consumption_path(year))
    upstream_consumption = utilities.clip_raster(
        cu_data,
        cu_profile,
        upstream_hucs.geometry,
        Path(pws_output_directory, f"{year}-cu-upstream.tif"),
    )

    wd_data, wd_profile, _ = read_national_raster(national_pws_withdrawal_path(year))
    downstream_withdrawal = utilities.clip_raster(
        wd_data,
        wd_profile,
        downstream_hucs.geometry,
        Path(pws_wd_output_directory, f"{year}-downstream.tif"),
    )

    return upstream_consumption, downstream_withdrawal


def prepare_withdrawal_raster_basin(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: Path,
    year: int,
) -> utilities.RasterResult:
    """
    Clips the surface-water public-supply withdrawal (prepare_pws_withdrawal_raster) to the basin HUCs.

    Writes pws-wd/<year>-basin.tif under root_output_directory.

    Parameters:
        hucs (geopandas.GeoDataFrame): The basin HUCs.
        root_output_directory (Path): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        result (RasterResult): The basin withdrawal, in mm/yr.
    """

    pws_wd_output_directory = utilities.create_directory(Path(root_output_directory, "pws-wd"))

    pws_wd, pws_wd_profile, _ = read_national_raster(national_pws_withdrawal_path(year))

    return utilities.clip_raster(
        pws_wd,
        pws_wd_profile,
        hucs.geometry,
        Path(pws_wd_output_directory, f"{year}-basin.tif"),
    )


def prepare_wrd_consumption_raster(
    hucs: geopandas.GeoDataFrame,
    root_output_directory: Path,
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
        root_output_directory (Path): The output directory of the basin.
        year (int): The year of the data.

    Returns:
        result (RasterResult): The basin consumption, on the irrigation grid, in mm/yr.
    """

    irr = prepare_irr_rasters(hucs, root_output_directory, year, region_tag="basin")

    pws_output_directory = utilities.create_directory(Path(root_output_directory, "pws"))
    cu_data, cu_profile, _ = read_national_raster(national_pws_consumption_path(year))
    pws_cu_basin = utilities.clip_raster(
        cu_data,
        cu_profile,
        hucs.geometry,
        Path(pws_output_directory, f"{year}-cu-basin.tif"),
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
        Path(root_output_directory, f"{year}-wrd-consumption-basin.tif"),
    )

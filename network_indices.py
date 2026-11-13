"""
RWY (Runoff Water Yield) and WRC (Water Resource Contribution) on the huc4, huc8 and
huc12 routing networks (LEVELS) — the downstream mirror of wrd_flows' WRD.

main.py's RWY/WRC split each basin once: every upstream cell against one scalar
averaged over a hand-picked downstream outlet region (e.g. huc4 1503 for the CRB).
Here each unit A gets its own downstream region instead: every unit its water passes
through on the way to its outlet (its descendants, wrd_flows.sum_descendants), found
from NHDPlus's HU12 tohuc routing (wrd_flows.build_flow_graph, ARTIFACT_EDGES
included). WRD pools over a unit's ancestors (what feeds it); RWY/WRC pool over its
descendants (what it feeds):

    RWY_cell = q_cell / mean(q over D_A)
    WRC_cell = (mean(Q over D_A) - mean(WW over D_A)) / (qi_cell - wc_cell)

    D_A = every unit downstream of the cell's unit A
    q   = GCM-averaged quickflow on its native 800 m grid (as main.run_rwy)
    Q   = the same quickflow as mm/yr on the 1 km water-use grid (as main.run_wrc)
    WW  = pws-sw withdrawal; qi, wc = the cell's runoff and consumption
          (irr + pws-cu + thermoelectric), exactly main.run_wrc's inputs

Means over D_A pool every cell of every descendant (main.py's safe_mean semantics:
finite, non-negative cells), not a mean of per-unit means. Units with nothing
downstream (outlets and closed-basin sinks) are left blank. There are no
hand-picked exclusions (e.g. the CRB's 1508): the routing decides.

Not called from main.main, since this module uses main's plotting; run it per basin
alongside wrd_flows.run, then conus_maps.run_network_rwy_wrc for the mosaic.
"""

from os import path

import geopandas
import numpy

import main
import raster_preparation
import utilities
import wrd_flows

LEVELS = (4, 8, 12)


def unit_sums(labels: numpy.ndarray, data: numpy.ndarray, n_labels: int) -> numpy.ndarray:
    """(sum, count) of finite, non-negative cells per label — utilities.safe_mean's
    cells — as an (n_labels, 2) array."""
    valid = numpy.isfinite(data) & (data >= 0)
    flat_labels = labels.ravel()
    return numpy.stack(
        [
            numpy.bincount(
                flat_labels, weights=numpy.where(valid, data, 0.0).ravel(), minlength=n_labels
            ),
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
    """Per label (row) and raster (column): the raster's mean over every cell of every
    unit downstream of that label's unit. NaN for label 0 (no unit), for units with
    nothing downstream, and where the descendants have no valid cells."""
    n_labels = len(unit_codes) + 1
    sums = numpy.concatenate([unit_sums(labels, r, n_labels) for r in rasters], axis=1)
    totals = wrd_flows.sum_descendants(
        graph, {code: sums[label] for label, code in enumerate(unit_codes, start=1)}
    )

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
    # Water-use rasters are NaN wherever there's no use, which means zero, not
    # missing data. Fill them with 0 inside the runoff footprint so WW is averaged
    # over the same cells as Q, and wc doesn't mask out cells.
    withdrawal = numpy.where(numpy.isfinite(runoff) & numpy.isnan(withdrawal), 0.0, withdrawal)
    consumption = numpy.where(numpy.isfinite(runoff) & numpy.isnan(consumption), 0.0, consumption)

    means = descendant_means(graph, unit_codes, labels, [runoff, withdrawal])
    numerator = (means[:, 0] - means[:, 1])[labels]

    denominator = runoff - consumption
    denominator[numpy.abs(denominator) < 1e-6] = numpy.nan

    with numpy.errstate(divide="ignore", invalid="ignore"):
        wrc = numerator / denominator
    wrc[~numpy.isfinite(runoff) | ~numpy.isfinite(consumption)] = numpy.nan
    wrc[~numpy.isfinite(wrc)] = numpy.nan
    return wrc


def run(
    huc2s: list[str],
    hucs: geopandas.GeoDataFrame,
    climate_models: list[str],
    years: list[int],
    root_output_directory: str,
    levels: list[int] = list(LEVELS),
) -> None:
    """Network RWY and WRC at every level in `levels`, for every climate model and
    year. `hucs` is the basin's huc4s (for clipping and map outlines), as for
    wrd_flows.run. The rasters are prepared once per year and shared by the levels."""
    hu12s = wrd_flows.load_hu12s(huc2s)
    networks = []
    for level in levels:
        units = wrd_flows.load_units(huc2s, level, hu12s)
        networks.append(
            {
                "level": level,
                "graph": wrd_flows.build_flow_graph(
                    wrd_flows.load_unit_destinations(hu12s, level), level
                ),
                "units": units,
                "unit_codes": units["unit"].tolist(),
            }
        )

    for climate_model in climate_models:
        for year in years:
            averaged, averaged_profile, _ = raster_preparation.average_quickflow_raster(
                climate_model, year, root_output_directory
            )
            quickflow, quickflow_profile, _ = utilities.clip_raster(
                averaged,
                averaged_profile,
                hucs.geometry,
                path.join(
                    root_output_directory,
                    "quickflow",
                    climate_model,
                    f"{climate_model}_{year}_averaged_basin.tif",
                ),
            )
            runoff, consumption, withdrawal, profile = wrd_flows.prepare_basin_rasters(
                hucs, climate_model, year, root_output_directory
            )

            for network in networks:
                level = network["level"]
                rwy_directory = utilities.create_directory(
                    path.join(root_output_directory, "rwy", f"huc{level}", climate_model)
                )
                wrc_directory = utilities.create_directory(
                    path.join(root_output_directory, "wrc", f"huc{level}", climate_model)
                )

                rwy = compute_rwy(
                    quickflow.astype("float64"),
                    wrd_flows.rasterize_units(network["units"], quickflow_profile),
                    network["unit_codes"],
                    network["graph"],
                )
                rwy, rwy_profile, _ = utilities.save_raster(
                    rwy,
                    quickflow_profile,
                    path.join(rwy_directory, f"{climate_model}_{year}_rwy.tif"),
                )
                utilities.compute_stats(rwy, f"RWY huc{level} ({climate_model}, {year})")
                main.plot_rwy_classes(
                    rwy,
                    rwy_profile,
                    hucs,
                    path.join(rwy_directory, f"{climate_model}_{year}_rwy.png"),
                    title=f"RWY huc{level} ({climate_model}): {year}",
                )

                wrc = compute_wrc(
                    runoff,
                    consumption,
                    withdrawal,
                    wrd_flows.rasterize_units(network["units"], profile),
                    network["unit_codes"],
                    network["graph"],
                )
                wrc, wrc_profile, _ = utilities.save_raster(
                    wrc,
                    profile,
                    path.join(wrc_directory, f"{climate_model}_{year}_wrc.tif"),
                )
                utilities.compute_stats(wrc, f"WRC huc{level} ({climate_model}, {year})")
                main.plot_wrc_classes(
                    wrc,
                    wrc_profile,
                    hucs,
                    path.join(wrc_directory, f"{climate_model}_{year}_wrc.png"),
                    title=f"WRC huc{level} ({climate_model}): {year}",
                )

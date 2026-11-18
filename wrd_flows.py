"""
Computes WRD (Water Resource Dependency) for each hydrologic unit of a basin, at the huc4, huc8 and huc12 levels.

The flow graph comes from the huc12 tohuc routing of the NHDPlus HR GDB. At huc12 the graph is the huc12 routing. At
huc4 and huc8 the graph uses the routing truncated to the unit code. Thus unit A has an edge to unit B if a huc12 in A
routes to a huc12 in B. The ancestors of a unit are all the units that drain into it, directly or through other units
of the graph. Each unit L gets:

    WRD_L = b_L / (Σb_M / N_L)

    b_L  = qd_L - ww_L        runoff minus withdrawal, for each cell of L
    Σb_M = Σqi_A - Σconsi_A   runoff minus consumption, summed over all cells of all ancestors A of L
    N_L  = number of cells of L with a value of b_L

A cell with no runoff value has no b_L. A cell with a runoff value and no withdrawal value has a withdrawal of 0
(raster_preparation.prepare_pws_withdrawal_raster). In the sums of Σb_M, a cell with no value adds nothing. A unit
without ancestors (a headwater) gets no WRD. WRD is 0 in a unit where Σb_M is not positive.

Σb_M does not decrease along the network. It sums the runoff minus the consumption of each ancestor, and does not
subtract the withdrawals of the units between an ancestor and L. Thus Σb_M increases downstream, and the terminal unit
gets the surplus of the full basin. This is intentional (decision of the project hydrologist). Do not change it to a
step-by-step depletion.

Known limitations:

- The model does not include inter-basin transfers. The consumption is where people use the water, not where the
  water leaves the river. Thus a unit that imports water looks like it consumes its own supply (for example the Gila
  huc4s, through the Central Arizona Project). An export out of the basin is not a depletion (for example the
  All-American Canal, the Colorado River Aqueduct, the Front Range transmountain diversions). The same applies to WRC
  in main.py.
- A unit that drains to more than one downstream unit adds its full surplus to each of them (build_flow_graph). The
  model does not divide a diverging flow between its branches.
- Inflow from outside the US is not in Σb_M, because the Quickflow runoff has no cells in Canada or Mexico. Thus a US
  unit downstream of such an area (for example the St. Lawrence, the Rio Grande below the Rio Conchos, the Columbia)
  looks more dependent than it is.
"""

import collections
import itertools
from pathlib import Path
from typing import Any, Literal, overload

import geopandas
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import pandas
from rasterio import features

import raster_preparation
import utilities

LEVELS = (4, 8, 12)

BasinRasters = tuple[
    numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[str, Any]
]  # (runoff, consumption, withdrawal, profile)

NHDPLUS_GDB = Path("input", "NHDPlus_H_National_Release_2_GDB.gdb")

# WRD classes of the notebook legend, with a band around 0 for the class "= 0". The colours are in the reverse order
# of the notebook, to complement WRC
WRD_EPSILON = 1e-6
WRD_BOUNDARIES = [-100, -2, -1, -WRD_EPSILON, WRD_EPSILON, 1, 2, 100]
WRD_COLORS = ["blue", "deepskyblue", "lightblue", "lightgray", "yellow", "orange", "darkred"]
WRD_TICKS = [(low + high) / 2 for low, high in itertools.pairwise(WRD_BOUNDARIES)]
WRD_TICK_LABELS = ["< −2", "−2 to −1", "−1 to 0", "= 0", "0 to 1", "1 to 2", "≥ 2"]

# ── Network ──────────────────────────────────────────────────────────────────
# Builds the unit-to-unit flow graph from the huc12 tohuc routing, and its headwaters and ancestor sets.

# tohuc values that are in no GDB huc12, and the GDB huc12 that gets the water. Checked by hand against the huc12 names
# and neighbours. The Great Lakes values are lake codes of another WBD version. Without a replacement, the huc12 is an
# outlet, and the network stops at the lake shore. Other values that are in no GDB huc12 stay outlets. They are the
# coastal bays and sounds of huc2 01, 02 and 03, which drain to the sea
TOHUC_REPLACEMENTS = {
    # the frontal huc12s of each lake, to the lake huc12
    "041800000300": "041800000200",  # Lake Superior
    "040602000000": "041900000200",  # Lake Michigan (St. Joseph, Kalamazoo and Muskegon rivers)
    "041900000000": "041900000200",  # Lake Michigan
    "041900000002": "041900000200",  # Lake Michigan (Green Bay)
    "040803000300": "042400020200",  # Lake Huron
    "042400000000": "042400020200",  # Lake Huron (Lake Michigan, Cheboygan River)
    "042400000200": "042400020200",  # Lake Huron (St. Marys River)
    "042400000300": "042400020200",  # Lake Huron
    "041202000300": "042600000200",  # Lake Erie
    "042600000300": "042600000200",  # Lake Erie
    "041502000200": "042800020200",  # Lake Ontario
    "042800000200": "042800020200",  # Lake Ontario (Trent River)
    "042800020100": "042800020200",  # Lake Ontario (Braddock Bay)
    "041503090000": "042900010407",  # from Lake Ontario to Wheeler Creek-Frontal Saint Lawrence River
    # one digit different from the downstream neighbour on the same river
    "010100090601": "010100090801",  # Gagnon Brook to Thibodeau Brook-Saint John River
    "010100200502": "010100100502",  # Lower Becaguimec Stream to Hales Brook-Saint John River
    "010300030603": "010300010603",  # Moosehead Lake to Indian Pond (Kennebec River)
    "042000020607": "042000020606",  # Echo Lake and Bar Creek to Echo Bay-Lower Echo River
}

# huc12s with a non-numeric tohuc that is not their outlet: (huc12, the GDB huc12 that gets the water)
TOHUC_OVERRIDES = {
    "041800000200": "042000020203",  # Lake Superior (tohuc CANADA) to City of Sault Sainte Marie-Saint Marys River
}


@overload
def load_huc12s(huc2s: list[str], ignore_geometry: Literal[False] = False) -> geopandas.GeoDataFrame: ...
@overload
def load_huc12s(huc2s: list[str], ignore_geometry: Literal[True]) -> pandas.DataFrame: ...
def load_huc12s(huc2s: list[str], ignore_geometry: bool = False) -> geopandas.GeoDataFrame | pandas.DataFrame:
    """
    Reads the WBDHU12 layer of the NHDPlus HR GDB, with the tohuc routing, for the given huc2s.

    The huc12 codes of the GDB and of the input/HUC shapefiles are from different WBD versions. Thus the unit polygons
    of all levels also come from the GDB (load_units).

    Corrects the tohuc with TOHUC_REPLACEMENTS and TOHUC_OVERRIDES. Prints the number of outlet huc12s with a numeric
    tohuc that is not a huc12 of the huc2s. Such a tohuc is a flow into another basin, or a code that is in no GDB
    huc12.

    Parameters:
        huc2s (list[str]): The huc2 codes.
        ignore_geometry (bool = False): Reads only the attributes when True.

    Returns:
        huc12s (geopandas.GeoDataFrame | pandas.DataFrame): One row for each huc12, with the columns huc12 and tohuc
            (strings). A DataFrame when ignore_geometry is True.
    """

    where = " OR ".join(f"huc12 LIKE '{huc2}%'" for huc2 in huc2s)
    huc12s = geopandas.read_file(NHDPLUS_GDB, layer="WBDHU12", where=where, ignore_geometry=ignore_geometry)

    huc12s["tohuc"] = huc12s["tohuc"].replace(TOHUC_REPLACEMENTS)
    for huc12, tohuc in TOHUC_OVERRIDES.items():
        huc12s.loc[huc12s["huc12"] == huc12, "tohuc"] = tohuc

    n_outside = int((huc12s["tohuc"].str.isdigit() & ~huc12s["tohuc"].isin(huc12s["huc12"])).sum())
    print(f"{n_outside} huc12s of {huc2s} route to a huc12 outside them: they are outlets")
    return huc12s


def load_unit_destinations(huc12s: pandas.DataFrame, level: int) -> pandas.DataFrame:
    """
    Counts the destinations of the huc12s of each unit, at the given level.

    A numeric tohuc becomes its unit code (the first level digits). A non-numeric tohuc (for example CLOSED BASIN or
    OCEAN) does not change. The count does not include a destination in the unit itself.

    Parameters:
        huc12s (pandas.DataFrame): The huc12s (load_huc12s).
        level (int): The number of digits of the unit code (4, 8 or 12).

    Returns:
        unit_destinations (pandas.DataFrame): One row for each unit, with the columns unit and destinations, a list of
            (destination, number of huc12s) from the most to the least frequent.
    """

    huc12s = huc12s[["huc12", "tohuc"]].copy()
    huc12s["unit"] = huc12s["huc12"].str[:level]
    huc12s["tounit"] = huc12s["tohuc"].mask(huc12s["tohuc"].str.isdigit(), huc12s["tohuc"].str[:level])

    return (
        huc12s.groupby("unit")["tounit"]
        .apply(lambda x: list(x[x != x.name].value_counts().items()))
        .reset_index()
        .rename(columns={"tounit": "destinations"})
    )


# (from unit, to unit) edges of the huc12 tohuc routing that are not real flow between the units, checked by hand
# against their huc12s. build_flow_graph removes them, because each one makes a full basin an ancestor of a neighbour.
# The codes are prefixes: a huc4 pair also removes each huc8 edge across it, and a huc8 pair applies only at huc8.
#
# Not applied at huc12, where each edge is the routing of one huc12: none of them makes a huc12 cycle, and a removal can
# cut the network. The 0706 -> 0707 huc12 (Prairie du Chien) is the only path downstream of the Upper Mississippi: it
# routes through the last huc12 of the Wisconsin and back into 0706. A removal disconnects the Upper Mississippi huc12s
# from all units below.
ARTIFACT_EDGES = {
    # the main stem of 0809 goes to the Gulf through West Bay and Head of Passes. The 0808 edge is two Terrebonne marsh
    # huc12s (Creole Bayou, Fourleague Bay). The 0317 edge is Lake Borgne into Mississippi Sound. With them, 0808 is the
    # only MRB outlet, and all ancestors of 0809 drain into it
    ("0809", "0808"),
    ("0809", "0317"),
    # the Wisconsin River (0707) flows into the Mississippi (0706) at Prairie du Chien. The reverse edge is the Prairie
    # du Chien Mississippi huc12, which routes into the last huc12 of the Wisconsin. With it, 0706 <-> 0707 is a cycle.
    # Then 0707 gets all of the Upper Mississippi as ancestors
    ("0706", "0707"),
    # the Pecos main stem is 1306 -> 1307 -> 1304. The reverse edge is one huc12, Gyp Draw-Delaware River, in 1307. But
    # the Delaware flows into the Pecos in the Coad Tank-Pecos River huc12 of 1306. With it, 1306 <-> 1307 is a cycle
    ("1307", "1306"),
    # the Beaver River (16030008) ends in the Sevier River (16030005) above Sevier Lake. The reverse edge is one huc12,
    # Pine Creek, which routes into Devils Kitchen Reservoir, a closed sink in 16030008. With it, 16030005 <-> 16030008
    # is a cycle
    ("16030005", "16030008"),
}


def is_artifact_edge(unit: str, dest: str, level: int) -> bool:
    """
    Finds if an edge is in ARTIFACT_EDGES at the given level.

    Parameters:
        unit (str): The code of the upstream unit.
        dest (str): The code of the downstream unit.
        level (int): The number of digits of the unit codes.

    Returns:
        is_artifact (bool): True if build_flow_graph removes the edge.
    """

    return level < 12 and any(
        len(from_code) <= level and unit.startswith(from_code) and dest.startswith(to_code)
        for from_code, to_code in ARTIFACT_EDGES
    )


def build_flow_graph(unit_destinations: pandas.DataFrame, level: int) -> dict[str, list[str]]:
    """
    Builds the downstream edges of each unit from its destination counts.

    Most huc12s of a unit drain to a sink in the unit (CLOSED BASIN), not out of it. Thus the most frequent destination
    is not the outlet. Each numeric destination (a unit code) that is not in ARTIFACT_EDGES is an edge. A unit with more
    than one edge (for example a delta distributary) is a full ancestor of each downstream unit. Its contribution is
    not divided between them. Prints a review message for each split with more than one huc12. Such a split is more
    probably a real bifurcation than one huc12 on a boundary.

    Parameters:
        unit_destinations (pandas.DataFrame): The destination counts (load_unit_destinations).
        level (int): The number of digits of the unit codes.

    Returns:
        graph (dict[str, list[str]]): The downstream units of each unit. An empty list for an outlet.
    """

    graph: dict[str, list[str]] = {}

    for _, row in unit_destinations.iterrows():
        unit = row["unit"]
        numeric_destinations = [
            (dest, count)
            for dest, count in row["destinations"]
            if dest.isdigit() and not is_artifact_edge(unit, dest, level)
        ]

        if len(numeric_destinations) > 1:
            if numeric_destinations[1][1] > 1:
                print(
                    f"  REVIEW: {unit} has multiple downstream units with "
                    f"substantial huc12 support {numeric_destinations} — verify "
                    "this is a genuine flow split, not a data artifact"
                )
            else:
                print(
                    f"  Note: {unit} has multiple candidate downstream units {numeric_destinations}, using all of them"
                )

        graph[unit] = [dest for dest, _count in numeric_destinations]

    return graph


def find_origins(graph: dict[str, list[str]]) -> list[str]:
    """
    Finds the headwater units: the units that are not a downstream unit of another unit.

    Parameters:
        graph (dict[str, list[str]]): The flow graph (build_flow_graph).

    Returns:
        origins (list[str]): The headwater units, in the order of the graph.
    """

    downstream_targets = {dest for dests in graph.values() for dest in dests}
    return [unit for unit in graph if unit not in downstream_targets]


def build_ancestor_map(graph: dict[str, list[str]]) -> dict[str, set[str]]:
    """
    Finds the ancestors of each unit: all the units that drain into it, directly or through other units.

    A unit with more than one downstream unit is a full ancestor of each of them.

    Parameters:
        graph (dict[str, list[str]]): The flow graph (build_flow_graph).

    Returns:
        ancestor_map (dict[str, set[str]]): The ancestors of each unit of the graph, without the unit itself.

    Raises:
        ValueError: If the graph has a cycle. A unit on a cycle counts its own runoff and consumption in its Σb_M.
    """

    children: dict[str, list[str]] = collections.defaultdict(list)
    for unit, downstreams in graph.items():
        for downstream in downstreams:
            children[downstream].append(unit)

    ancestor_map: dict[str, set[str]] = {}
    for unit in graph:
        ancestors: set[str] = set()
        stack = list(children.get(unit, []))
        while stack:
            upstream_unit = stack.pop()
            if upstream_unit in ancestors:
                continue
            ancestors.add(upstream_unit)
            stack.extend(children.get(upstream_unit, []))
        if unit in ancestors:
            raise ValueError(
                f"Cycle in flow graph: {unit} drains into itself. Find the bogus "
                "edge with print_flow_chains and add it to ARTIFACT_EDGES."
            )
        ancestor_map[unit] = ancestors

    return ancestor_map


def sum_descendants(graph: dict[str, list[str]], unit_totals: dict[str, numpy.ndarray]) -> dict[str, numpy.ndarray]:
    """
    Sums the totals of the descendants of each unit: all the units downstream of it, without the unit itself.

    When no unit has more than one downstream unit (always at huc12), one walk with memoization gives all the sums. When
    a unit has more than one, the walk counts a split that joins again two times. Then the descendant sets come from
    build_ancestor_map, and the sum includes a unit downstream of both branches one time.

    Parameters:
        graph (dict[str, list[str]]): The flow graph (build_flow_graph).
        unit_totals (dict[str, numpy.ndarray]): The totals of each unit, all with the same shape. A unit that is not in
            it (no polygon in the basin) adds zeros.

    Returns:
        totals (dict[str, numpy.ndarray]): The sum over the descendants of each unit.

    Raises:
        ValueError: If the graph has a cycle.
    """

    zero = numpy.zeros_like(next(iter(unit_totals.values())))

    if any(len(downstreams) > 1 for downstreams in graph.values()):
        descendant_map: dict[str, set[str]] = {unit: set() for unit in graph}
        for unit, ancestors in build_ancestor_map(graph).items():
            for ancestor in ancestors:
                descendant_map[ancestor].add(unit)
        # sorted, because the order of a set of strings changes in each process, and the float sum with it
        return {
            unit: sum((unit_totals.get(d, zero) for d in sorted(descendants)), zero)
            for unit, descendants in descendant_map.items()
        }

    totals: dict[str, numpy.ndarray] = {}
    for start in graph:
        # walk down to a unit with a known total (or an outlet), then fill in the chain from the bottom up
        chain: list[str] = []
        on_chain: set[str] = set()
        unit = start
        while unit not in totals and graph.get(unit):
            if unit in on_chain:
                raise ValueError(f"Cycle in flow graph: {unit} drains into itself.")
            chain.append(unit)
            on_chain.add(unit)
            unit = graph[unit][0]
        if unit not in totals:
            totals[unit] = zero
        for upstream_unit in reversed(chain):
            downstream = graph[upstream_unit][0]
            totals[upstream_unit] = totals[downstream] + unit_totals.get(downstream, zero)

    return totals


def trace_paths(origin: str, graph: dict[str, list[str]]) -> list[list[str]]:
    """
    Finds all downstream chains from a headwater to an outlet, with a branch at each unit with more than one downstream
    unit. For print_flow_chains only.

    Parameters:
        origin (str): The headwater unit.
        graph (dict[str, list[str]]): The flow graph (build_flow_graph).

    Returns:
        paths (list[list[str]]): The chains, each from the origin to an outlet.

    Raises:
        ValueError: If a chain has a cycle.
    """

    def walk(unit: str, visited: frozenset[str]) -> list[list[str]]:
        if unit in visited:
            raise ValueError(f"Cycle detected in flow graph at {unit}")
        visited = visited | {unit}
        downstreams = graph.get(unit, [])
        if not downstreams:
            return [[unit]]
        return [[unit] + rest for downstream in downstreams for rest in walk(downstream, visited)]

    return walk(origin, frozenset())


def print_flow_chains(huc2s: list[str], level: int = 4) -> None:
    """
    Prints the downstream chains of each headwater to its outlets. A diagnostic, not a pipeline step. Use it to check
    the network, for example before a change to ARTIFACT_EDGES or to the upstream/downstream huc4s of
    main.BASIN_MAPPINGS.

    Parameters:
        huc2s (list[str]): The huc2 codes.
        level (int = 4): The number of digits of the unit codes.
    """

    unit_destinations = load_unit_destinations(load_huc12s(huc2s, ignore_geometry=True), level)
    print(unit_destinations)

    graph = build_flow_graph(unit_destinations, level)
    origins = find_origins(graph)

    print(f"\n{len(origins)} origin(s) found: {origins}\n")
    for origin in origins:
        paths = trace_paths(origin, graph)
        if len(paths) == 1:
            print(" → ".join(paths[0]))
        else:
            print(f"{origin} branches into {len(paths)} paths:")
            for branch_path in paths:
                print("  " + " → ".join(branch_path))


# ── Units ─────────────────────────────────────────────────────────────────────


def load_units(huc12s: geopandas.GeoDataFrame, level: int) -> geopandas.GeoDataFrame:
    """
    Makes the unit polygons of the given level from the huc12s, so that the polygons and the flow graph have the same
    unit codes.

    huc4 and huc8 are the huc12s dissolved by code prefix. The GDB huc12s are not a valid coverage, thus the dissolve is
    a full union, not a coverage union.

    Parameters:
        huc12s (geopandas.GeoDataFrame): The huc12s (load_huc12s).
        level (int): The number of digits of the unit codes.

    Returns:
        units (geopandas.GeoDataFrame): One row for each polygon, with the columns unit and geometry.
    """

    units = huc12s[["huc12", "geometry"]].rename(columns={"huc12": "unit"})
    if level < 12:
        units["unit"] = units["unit"].str[:level]
        units = units.dissolve(by="unit", method="unary").reset_index()
    return geopandas.GeoDataFrame(units[["unit", "geometry"]], geometry="geometry").reset_index(drop=True)


def rasterize_units(units: geopandas.GeoDataFrame, profile: dict[str, Any]) -> numpy.ndarray:
    """
    Labels each cell of the grid with its unit.

    A cell belongs to the unit that contains its centre (all_touched=False, as utilities.clip_raster). Thus a unit and
    its ancestors never share a cell.

    Parameters:
        units (geopandas.GeoDataFrame): The unit polygons (load_units), in any CRS.
        profile (dict[str, Any]): The profile of the grid.

    Returns:
        labels (numpy.ndarray): int32, 1 + the row of the unit in units, 0 for a cell outside all units.
    """

    units_in_crs = units.to_crs(profile["crs"])
    return features.rasterize(
        zip(units_in_crs.geometry, range(1, len(units_in_crs) + 1), strict=True),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        all_touched=False,
        dtype="int32",
    )


# ── Per-unit WRD ──────────────────────────────────────────────────────────────


def compute_unit_wrd(
    labels: numpy.ndarray,
    unit_codes: list[str],
    ancestor_map: dict[str, set[str]],
    runoff: numpy.ndarray,
    consumption: numpy.ndarray,
    withdrawal: numpy.ndarray,
) -> tuple[numpy.ndarray, pandas.DataFrame]:
    """
    Computes the WRD of each unit with ancestors, against its own ancestors (see the module docstring).

    The sums for each unit come from one bincount over the labels. Σb_M is the sum of the ancestor sums. This is equal
    to a clip to the union of the ancestors, because the labels do not overlap.

    Parameters:
        labels (numpy.ndarray): The unit labels (rasterize_units).
        unit_codes (list[str]): The unit code of each label, from label 1.
        ancestor_map (dict[str, set[str]]): The ancestors of each unit (build_ancestor_map).
        runoff (numpy.ndarray): The runoff, in mm/yr, NaN for no data.
        consumption (numpy.ndarray): The consumption, in mm/yr, NaN for no data, on the same grid.
        withdrawal (numpy.ndarray): The withdrawal, in mm/yr, NaN for no data, on the same grid. NaN counts as 0 where
            the runoff has a value.

    Returns:
        wrd (numpy.ndarray): The WRD of each cell. NaN outside the units with a WRD.
        unit_summary (pandas.DataFrame): One row for each unit with a WRD, with the columns unit, n_ancestors,
            upstream_surplus (Σb_M) and n_cells (N_L).
    """

    n_labels = len(unit_codes) + 1
    flat_labels = labels.ravel()

    withdrawal = numpy.where(numpy.isfinite(runoff) & numpy.isnan(withdrawal), 0.0, withdrawal)
    b_L = runoff - withdrawal  # noqa: N806 (name of the formula)

    unit_surplus = numpy.bincount(
        flat_labels, weights=numpy.nan_to_num(runoff).ravel(), minlength=n_labels
    ) - numpy.bincount(flat_labels, weights=numpy.nan_to_num(consumption).ravel(), minlength=n_labels)
    n_cells = numpy.bincount(flat_labels, weights=(~numpy.isnan(b_L.ravel())).astype(float), minlength=n_labels).astype(
        int
    )
    n_cells[0] = 0  # label 0 is "no unit"

    label_of = {code: label for label, code in enumerate(unit_codes, start=1)}
    # for each label: the divisor Σb_M / N_L, NaN for a unit without WRD, infinite where Σb_M <= 0 (WRD = 0)
    divisor = numpy.full(n_labels, numpy.nan)
    rows = []
    for label, code in enumerate(unit_codes, start=1):
        # sorted, because the order of a set of strings changes in each process, and the float sum with it
        ancestor_labels = [label_of[a] for a in sorted(ancestor_map.get(code, ())) if a in label_of]
        if not ancestor_labels or n_cells[label] == 0:
            continue
        upstream_surplus = float(unit_surplus[ancestor_labels].sum())
        divisor[label] = upstream_surplus / n_cells[label] if upstream_surplus > 0 else numpy.inf
        rows.append(
            {
                "unit": code,
                "n_ancestors": len(ancestor_labels),
                "upstream_surplus": upstream_surplus,
                "n_cells": int(n_cells[label]),
            }
        )

    with numpy.errstate(divide="ignore", invalid="ignore"):
        wrd = b_L / divisor[labels]
    # b_L / inf is ±0.0. The class "= 0" must not show -0.0
    wrd = numpy.where(numpy.isinf(divisor[labels]) & ~numpy.isnan(b_L), 0.0, wrd)
    wrd[~numpy.isfinite(wrd)] = numpy.nan

    return wrd, pandas.DataFrame(rows, columns=["unit", "n_ancestors", "upstream_surplus", "n_cells"])


def unit_stats(wrd: numpy.ndarray, labels: numpy.ndarray, unit_codes: list[str]) -> pandas.DataFrame:
    """
    Computes the statistics of utilities.compute_stats for each unit.

    Parameters:
        wrd (numpy.ndarray): The WRD of each cell.
        labels (numpy.ndarray): The unit labels (rasterize_units).
        unit_codes (list[str]): The unit code of each label, from label 1.

    Returns:
        stats (pandas.DataFrame): One row for each unit with a finite cell, with the columns unit, min, q1, median,
            mean, q3 and max.
    """

    valid = numpy.isfinite(wrd) & (labels > 0)
    grouped = pandas.Series(wrd[valid]).groupby(numpy.asarray(unit_codes)[labels[valid] - 1])
    return (
        pandas.DataFrame(
            {
                "min": grouped.min(),
                "q1": grouped.quantile(0.25),
                "median": grouped.median(),
                "mean": grouped.mean(),
                "q3": grouped.quantile(0.75),
                "max": grouped.max(),
            }
        )
        .rename_axis("unit")
        .reset_index()
    )


# ── Orchestration ─────────────────────────────────────────────────────────────


def plot_wrd_cells(
    wrd: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plots the WRD of each cell in the WRD classes with utilities.plot_raster_map. Prints the number of cells in each
    class.

    Parameters:
        wrd (numpy.ndarray): The WRD of each cell.
        profile (dict[str, Any]): The profile of the grid.
        hucs (geopandas.GeoDataFrame): The HUCs whose outlines to plot, in any CRS.
        output_path (Path): The path to the output PNG.
        title (str): The map title.
    """

    # the class of each cell as BoundaryNorm gives it: [low, high), and the end classes are open
    finite = wrd[numpy.isfinite(wrd)]
    classes = numpy.bincount(numpy.digitize(finite, WRD_BOUNDARIES[1:-1]), minlength=len(WRD_COLORS))
    counts = {label: int(count) for label, count in zip(WRD_TICK_LABELS, classes, strict=True)}
    print(f"  WRD category counts: {counts}")

    utilities.plot_raster_map(
        wrd,
        profile,
        hucs,
        output_path,
        title=title,
        cmap=mcolors.ListedColormap(WRD_COLORS),
        norm=mcolors.BoundaryNorm(boundaries=WRD_BOUNDARIES, ncolors=len(WRD_COLORS)),
        cbar_label="WRD Categories",
        cbar_ticks=WRD_TICKS,
        cbar_ticklabels=WRD_TICK_LABELS,
    )


def plot_network_wrd(
    units: geopandas.GeoDataFrame,
    unit_summary: pandas.DataFrame,
    level: int,
    output_path: Path,
    title: str,
) -> None:
    """
    Plots each unit in the WRD class of its mean WRD, and saves it as a PNG. A unit without WRD is light grey.

    Parameters:
        units (geopandas.GeoDataFrame): The unit polygons (load_units).
        unit_summary (pandas.DataFrame): The per-unit summary, with the columns unit and mean.
        level (int): The number of digits of the unit codes.
        output_path (Path): The path to the output PNG.
        title (str): The map title.
    """

    plot_df = units.copy()
    plot_df["mean_wrd"] = plot_df["unit"].map(unit_summary.set_index("unit")["mean"])
    # thin outlines at the finer levels, else the outlines hide the fill
    linewidth = 0.3 if level == 4 else 0.05

    cmap = mcolors.ListedColormap(WRD_COLORS)
    norm = mcolors.BoundaryNorm(WRD_BOUNDARIES, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    has_value = plot_df["mean_wrd"].notna()
    if not has_value.all():
        plot_df[~has_value].plot(ax=ax, color="whitesmoke", edgecolor="black", linewidth=linewidth)
    if has_value.any():
        plot_df[has_value].plot(
            column="mean_wrd",
            cmap=cmap,
            norm=norm,
            ax=ax,
            edgecolor="black",
            linewidth=linewidth,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, ticks=WRD_TICKS)
    cbar.ax.set_yticklabels(WRD_TICK_LABELS)
    cbar.set_label(f"Mean WRD by huc{level}")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Network WRD map saved → {output_path}")


def build_level(huc12s: geopandas.GeoDataFrame, units: geopandas.GeoDataFrame, level: int) -> dict[str, Any]:
    """
    Builds the network of a level, which is the same for all scenarios and years. WRD (run_level) and the network RWY
    and WRC (network_indices.run_year) use it.

    Parameters:
        huc12s (geopandas.GeoDataFrame): The huc12s (load_huc12s).
        units (geopandas.GeoDataFrame): The unit polygons of the level (load_units).
        level (int): The number of digits of the unit codes.

    Returns:
        network (dict[str, Any]): The keys level, units, unit_codes, graph (build_flow_graph), ancestor_map
            (build_ancestor_map) and labels (for unit_labels). unit_codes gives the unit code of each label, from label 1.
    """

    graph = build_flow_graph(load_unit_destinations(huc12s, level), level)

    outlets = [unit for unit, downstream in graph.items() if not downstream]
    print(f"huc{level}: {len(graph)} units, {len(outlets)} outlet(s), {len(find_origins(graph))} headwater(s)")

    return {
        "level": level,
        "units": units,
        "unit_codes": units["unit"].tolist(),
        "graph": graph,
        "ancestor_map": build_ancestor_map(graph),
        "labels": {},
    }


def unit_labels(network: dict[str, Any], profile: dict[str, Any]) -> numpy.ndarray:
    """
    Gives the unit labels of a network on a grid (rasterize_units).

    The labels depend only on the level and the grid. Thus the network keeps the labels of each grid, and a later call
    on the same grid uses them again.

    Parameters:
        network (dict[str, Any]): The network of the level (build_level).
        profile (dict[str, Any]): The profile of the grid.

    Returns:
        labels (numpy.ndarray): The unit labels (rasterize_units).
    """

    grid = (str(profile["crs"]), profile["transform"], profile["height"], profile["width"])
    labels_by_grid: dict[tuple[Any, ...], numpy.ndarray] = network["labels"]
    if grid not in labels_by_grid:
        labels_by_grid[grid] = rasterize_units(network["units"], profile)
    return labels_by_grid[grid]


def prepare_water_use(
    hucs: geopandas.GeoDataFrame,
    year: int,
    root_output_directory: Path,
) -> tuple[utilities.RasterResult, utilities.RasterResult]:
    """
    Prepares the consumption and the withdrawal of the basin (raster_preparation). They do not depend on the scenario.

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons of the basin.
        year (int): The year.
        root_output_directory (Path): The output directory of the basin.

    Returns:
        consumption (RasterResult): The consumption (raster_preparation.prepare_wrd_consumption_raster), in mm/yr.
        withdrawal (RasterResult): The withdrawal (raster_preparation.prepare_withdrawal_raster_basin), in mm/yr.
    """

    consumption = raster_preparation.prepare_wrd_consumption_raster(hucs, root_output_directory, year)
    withdrawal = raster_preparation.prepare_withdrawal_raster_basin(hucs, root_output_directory, year)
    return consumption, withdrawal


def prepare_basin_rasters(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: Path,
    water_use: tuple[utilities.RasterResult, utilities.RasterResult],
) -> BasinRasters:
    """
    Prepares the runoff of the basin (raster_preparation.prepare_runoff_raster_basin), and gives it with the water use.

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons of the basin.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year.
        root_output_directory (Path): The output directory of the basin.
        water_use (tuple[RasterResult, RasterResult]): The consumption and the withdrawal of the year
            (prepare_water_use).

    Returns:
        rasters (BasinRasters): The runoff, the consumption and the withdrawal, float64, in mm/yr, NaN for no data, and
            the profile of the runoff grid.

    Raises:
        ValueError: If the consumption or the withdrawal is not on the runoff grid.
    """

    runoff, profile, _ = raster_preparation.prepare_runoff_raster_basin(
        hucs, climate_model, year, root_output_directory
    )
    (consumption, consumption_profile, consumption_path), (withdrawal, withdrawal_profile, withdrawal_path) = water_use

    for data, data_profile, data_path in (
        (consumption, consumption_profile, consumption_path),
        (withdrawal, withdrawal_profile, withdrawal_path),
    ):
        if data_profile["transform"] != profile["transform"] or data.shape != runoff.shape:
            raise ValueError(f"{data_path} isn't on the same grid as the runoff raster")

    return runoff.astype("float64"), consumption.astype("float64"), withdrawal.astype("float64"), profile


def run_year(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: Path,
    networks: list[dict[str, Any]],
    rasters: BasinRasters,
) -> None:
    """
    Computes and writes the WRD of one scenario and year at each level (run_level).

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons of the basin, for the map outlines.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year.
        root_output_directory (Path): The output directory of the basin.
        networks (list[dict[str, Any]]): The networks (build_level).
        rasters (BasinRasters): The rasters of the scenario and year (prepare_basin_rasters).
    """

    runoff, consumption, withdrawal, profile = rasters
    for network in networks:
        run_level(hucs, climate_model, year, root_output_directory, network, runoff, consumption, withdrawal, profile)


def run_level(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: Path,
    network: dict[str, Any],
    runoff: numpy.ndarray,
    consumption: numpy.ndarray,
    withdrawal: numpy.ndarray,
    profile: dict[str, Any],
) -> None:
    """
    Computes the WRD of one level, scenario and year, and writes it.

    Writes these files in wrd/huc<level>/<climate_model>/ under root_output_directory, each with the name prefix
    <climate_model>_<year>:

    - _wrd.tif and _wrd.png: the WRD of each cell.
    - _network.png: the mean WRD of each unit.
    - _summary.csv: one row for each unit with a WRD.

    A level where no unit has a WRD gets an all-NaN raster and an empty summary.

    Parameters:
        hucs (geopandas.GeoDataFrame): The huc4 polygons of the basin, for the map outlines.
        climate_model (str): The scenario, for example SSP245.
        year (int): The year.
        root_output_directory (Path): The output directory of the basin.
        network (dict[str, Any]): The network of the level (build_level).
        runoff (numpy.ndarray): The runoff (prepare_basin_rasters).
        consumption (numpy.ndarray): The consumption (prepare_basin_rasters).
        withdrawal (numpy.ndarray): The withdrawal (prepare_basin_rasters).
        profile (dict[str, Any]): The profile of the grid.
    """

    level = network["level"]
    units = network["units"]
    unit_codes = network["unit_codes"]

    out_dir = Path(root_output_directory, "wrd", f"huc{level}", climate_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{climate_model}_{year}"

    labels = unit_labels(network, profile)
    wrd, unit_summary = compute_unit_wrd(labels, unit_codes, network["ancestor_map"], runoff, consumption, withdrawal)
    unit_summary = unit_summary.merge(unit_stats(wrd, labels, unit_codes), on="unit", how="left")
    print(
        f"huc{level} ({climate_model}, {year}): WRD for {len(unit_summary)} of "
        f"{len(units)} units (the rest are headwaters or have no valid cells)"
    )
    if level == 4:
        for row in unit_summary.itertuples():
            print(f"{row.unit}: {row.n_ancestors} upstream huc4s, Σb_M={row.upstream_surplus:.2f}, N_L={row.n_cells}")

    utilities.save_raster(wrd, profile, out_dir / f"{prefix}_wrd.tif")
    title = f"WRD huc{level} ({climate_model}): {year}"
    utilities.print_stats(wrd, title)
    plot_wrd_cells(wrd, profile, hucs, out_dir / f"{prefix}_wrd.png", title)

    summary_path = out_dir / f"{prefix}_summary.csv"
    unit_summary.rename(columns={"unit": f"huc{level}"})[
        [f"huc{level}", "n_ancestors", "upstream_surplus", "n_cells", "min", "q1", "median", "mean", "q3", "max"]
    ].sort_values(f"huc{level}").to_csv(summary_path, index=False)
    print(f"Per-huc{level} summary saved → {summary_path}")

    plot_network_wrd(
        units,
        unit_summary,
        level,
        out_dir / f"{prefix}_network.png",
        f"Mean WRD by huc{level} ({climate_model}): {year}",
    )

"""
WRD (Water Resource Dependency), computed per hydrologic unit against that unit's
own true upstream drainage — instead of once for the whole basin. The unit can be a
huc4, huc8 or huc12 (LEVELS); everything below is written in terms of "unit".

The original notebook treats the entire basin as one binary split: every non-outlet
huc4 pooled into a single "upstream" surplus, evaluated against one hardcoded
"downstream" outlet. That answers "how dependent is the outlet on the whole
watershed" but can't say anything about the network's structure in between.

This module builds an NHDPlus-derived routing graph from HU12 "tohuc" attributes
(the Network section below) and uses it to find each unit's REAL upstream
drainage — every unit that transitively flows into it, found by walking the graph,
not by hop-distance or any other proxy (see run()'s comment on why hop-distance
alone is unsound: 22 of 84 huc4 edges in the MRB have a successor whose hop isn't
exactly one more than its predecessor's). At level 12 the graph is NHDPlus's own
HU12 routing; at coarser levels, HU12 routing is truncated to the unit's code, so an
edge exists wherever any HU12 in one unit routes into another. Each unit L is then
evaluated independently:

    WRD_L = b_L / (Σb_M / N_L)

    b_L  = qd_L - ww_L        L's own per-cell runoff minus its own withdrawal
    Σb_M = Σqi_A - Σconsi_A    pooled runoff minus consumption over every unit A that
                               TRULY, transitively drains into L (ancestor_map[L])
    N_L  = number of valid cells within L itself

Units with no ancestors (headwaters/origins) are skipped — Σb_M is trivially 0
there. Every other unit's independently-computed result is then bundled into
cumulative tier-N.tif files purely for output convenience: tier N covers every
unit from hop 1 through hop N (hop = shortest distance from the nearest headwater,
compute_depth_tiers), with each cell keeping the value computed for its own unit's
true ancestry above — tiering never changes what counts as upstream, only which
file a result lands in. Only huc4 writes every tier; finer levels have hundreds of
hops, so they write just the final (full) tier.

Water is NOT progressively depleted down the chain, and this is intentional (per the
project hydrologist): Σb_M pools every ancestor's *raw* production (runoff minus its
own consumption) and never nets out withdrawals an intermediate unit made before
that water reaches L. A unit's contributing area only grows as you move downstream,
so the further down the network you go, the larger Σb_M gets — the terminal unit
has the whole basin's surplus available to it. Do not "fix" this into step-by-step
depletion.

Known limitation: inter-basin transfers are not modeled. Consumption is located
where water is used, not where it's diverted, so a unit that imports water (e.g.
the Gila huc4s via the Central Arizona Project) looks like it consumes its own
supply, and exports out of the basin (e.g. the All-American Canal, Colorado River
Aqueduct, Front Range transmountain diversions) never show up as depletion at
all. The same applies to WRC in main.py.
"""

import collections
import os
from os import path
from typing import Any

import geopandas
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import pandas
import rasterio
from rasterio import features

import raster_preparation
import utilities

LEVELS = (4, 8, 12)

# ── Network ──────────────────────────────────────────────────────────────────
# Builds the unit-to-unit flow graph from NHDPlus's HU12 "tohuc" routing, and
# everything derived from it: origins, true ancestor sets, hop-distance tiers.


def load_hu12s(huc2s: list[str], ignore_geometry: bool = False) -> pandas.DataFrame:
    """NHDPlus WBD HU12s (with their tohuc routing) for every huc2 in `huc2s`. The
    GDB's HU12 codes don't all match input/HUC's WBDHU12 shapefiles (different WBD
    vintages), so level-12 geometry comes from here too, alongside its routing."""
    where = " OR ".join(f"huc12 LIKE '{huc2}%'" for huc2 in huc2s)
    return geopandas.read_file(
        path.join("input", "NHDPlus_H_National_Release_2_GDB.gdb"),
        layer="WBDHU12",
        where=where,
        ignore_geometry=ignore_geometry,
    )


def load_unit_destinations(hu12s: pandas.DataFrame, level: int) -> pandas.DataFrame:
    hu12s = hu12s[["huc12", "tohuc"]].copy()
    hu12s["unit"] = hu12s["huc12"].str[:level]
    hu12s["tounit"] = hu12s["tohuc"].where(
        ~(hu12s["tohuc"].str.isdigit() & (hu12s["tohuc"].str.len() >= level)),
        other=hu12s["tohuc"].str[:level],
    )

    return (
        hu12s.groupby("unit")["tounit"]
        .apply(lambda x: list(x[x != x.name].value_counts().items()))
        .reset_index()
        .rename(columns={"tounit": "destinations"})
    )


# (from unit, to unit) edges that NHDPlus's HU12 tohuc routing produces but that
# aren't real unit-to-unit flow, checked by hand against the HU12s behind them.
# Dropped in build_flow_graph so they don't make a whole basin an ancestor of a
# neighbor. Codes are prefixes: a huc4 pair also drops every huc8 edge crossing it,
# and a huc8 pair applies only at huc8.
#
# Never applied at huc12, where each edge is just one HU12's own routing: none of
# these make a huc12 cycle, and dropping one can cut the network instead. The
# 0706 -> 0707 HU12 (Prairie du Chien, 0 km²) is the Upper Mississippi's only path
# downstream — it routes through the Wisconsin's last HU12 (also 0 km²) and back
# into 0706 — so dropping it would orphan ~2,000 HU12s from everything below.
ARTIFACT_EDGES = {
    # 0809's main stem leaves via West Bay/Head of Passes to the Gulf. The 0808 edge
    # is two Terrebonne marsh HU12s (Creole Bayou, Fourleague Bay; ~215 km²), and
    # the 0317 edge is Lake Borgne spilling into Mississippi Sound. Keeping them made
    # 0808 the MRB's only outlet, with all 77 of 0809's ancestors draining into it.
    ("0809", "0808"),
    ("0809", "0317"),
    # The Wisconsin River (0707) joins the Mississippi (0706) at Prairie du Chien;
    # the reverse edge comes from the Prairie du Chien Mississippi HU12 being routed
    # into the Wisconsin's last HU12. Keeping it made a 0706 <-> 0707 cycle, giving
    # 0707 all of the Upper Mississippi as ancestors.
    ("0706", "0707"),
    # The Pecos main stem runs 1306 -> 1307 -> 1304. The reverse edge is one HU12,
    # Gyp Draw-Delaware River (~157 km²): it's in 1307 but the Delaware joins the
    # Pecos just inside 1306's Coad Tank-Pecos River HU12. Keeping it made a
    # 1306 <-> 1307 cycle.
    ("1307", "1306"),
    # The Beaver River (16030008) ends in the Sevier River (16030005) just above
    # Sevier Lake. The reverse edge is one HU12, Pine Creek (~157 km²), routed into
    # Devils Kitchen Reservoir, a closed sink in 16030008 that drains nowhere.
    # Keeping it made a 16030005 <-> 16030008 cycle.
    ("16030005", "16030008"),
}


def is_artifact_edge(unit: str, dest: str, level: int) -> bool:
    return level < 12 and any(
        len(from_code) <= level and unit.startswith(from_code) and dest.startswith(to_code)
        for from_code, to_code in ARTIFACT_EDGES
    )


def build_flow_graph(unit_destinations: pandas.DataFrame, level: int) -> dict[str, list[str]]:
    """Reduce each unit's raw tounit tallies to its real downstream edge(s).

    Most HU12s within a unit drain to internal sinks ("CLOSED BASIN") rather than
    exiting the unit, so the majority destination is usually noise, not signal. The
    unit as a whole only has an outlet if at least one HU12 in it actually routes to
    another unit — so the real downstream edges are whichever destinations are a
    unit code (numeric), not whichever destination is most common.

    A unit can legitimately drain to more than one downstream unit (e.g. a delta
    distributary, or a HU12 straddling a boundary), so all numeric destinations are
    kept as separate edges rather than arbitrarily picking one — each is treated as
    a full ancestor of whichever downstream unit(s) it feeds (see
    build_ancestor_map), with no attempt to apportion its contribution between
    them. A split backed by more than one HU12 is flagged for manual review, since
    that's more likely a genuine bifurcation than a single stray boundary HU12.
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
                    f"substantial HU12 support {numeric_destinations} — verify "
                    "this is a genuine flow split, not a data artifact"
                )
            else:
                print(
                    f"  Note: {unit} has multiple candidate downstream units {numeric_destinations}, using all of them"
                )

        graph[unit] = [dest for dest, _count in numeric_destinations]

    return graph


def find_origins(graph: dict[str, list[str]]) -> list[str]:
    """Headwater units: those that are never anyone else's downstream destination."""
    downstream_targets = {dest for dests in graph.values() for dest in dests}
    return [unit for unit in graph if unit not in downstream_targets]


def build_ancestor_map(graph: dict[str, list[str]]) -> dict[str, set[str]]:
    """For every unit, the set of units that transitively drain into it (excludes itself).

    `graph` is a DAG: most units have exactly one downstream edge, but a unit whose
    HU12s split across a boundary can have more than one (see build_flow_graph),
    and any number of upstream tributaries can point at the same downstream unit.
    This inverts that into a children map and walks it from every node to collect
    its full upstream drainage — not just its immediate neighbor(s). A unit that
    feeds more than one downstream unit becomes a full ancestor of each of them;
    its contribution is not split between branches.

    Raises ValueError if the graph has a cycle: a unit on one would count its own
    runoff and consumption in its Σb_M.
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
    """For every unit, `unit_totals` summed over every unit downstream of it (its
    descendants — excludes itself): the mirror of build_ancestor_map, pooled.

    When every unit has at most one downstream edge — always at huc12, where each
    HU12 has a single tohuc — a unit's descendants are just its downstream neighbor
    plus that neighbor's descendants, so each total comes from one memoized walk
    instead of materializing every descendant set (which at huc12 would be tens of
    millions of entries in the MRB). Coarser levels can split (see
    build_flow_graph), and a split that later rejoins would be counted twice by the
    walk, so there every descendant set is built explicitly from the ancestor map
    instead — cheap at a few hundred units. A unit downstream of both branches is
    counted once. Units missing from `unit_totals` (no polygon in this basin)
    contribute zeros."""
    zero = numpy.zeros_like(next(iter(unit_totals.values())))

    if any(len(downstreams) > 1 for downstreams in graph.values()):
        descendant_map: dict[str, set[str]] = {unit: set() for unit in graph}
        for unit, ancestors in build_ancestor_map(graph).items():
            for ancestor in ancestors:
                descendant_map[ancestor].add(unit)
        return {
            unit: sum((unit_totals.get(d, zero) for d in descendants), zero)
            for unit, descendants in descendant_map.items()
        }

    totals: dict[str, numpy.ndarray] = {}
    for start in graph:
        # Walk down until reaching a unit whose total is already known (or an outlet),
        # then fill the chain back in from the bottom up.
        chain = []
        on_chain = set()
        unit = start
        while unit not in totals and graph.get(unit):
            if unit in on_chain:
                raise ValueError(f"Cycle in flow graph: {unit} drains into itself.")
            chain.append(unit)
            on_chain.add(unit)
            unit = graph[unit][0]
        if unit not in totals:  # an outlet: nothing below it
            totals[unit] = zero
        for upstream_unit in reversed(chain):
            downstream = graph[upstream_unit][0]
            totals[upstream_unit] = totals[downstream] + unit_totals.get(downstream, zero)

    return totals


def compute_depth_tiers(graph: dict[str, list[str]]) -> dict[str, int]:
    """Shortest hop-distance from the nearest origin (headwater), walking downstream.

    Origins get depth 0. When multiple origins (or multiple branches) can reach the
    same downstream unit via chains of different lengths, the unit's depth is the
    length of the *shortest* such chain — e.g. in the MRB, huc4 0801 is 2 hops from
    its nearest origin (via 0514), however far away its other origins are. This is
    a plain multi-source BFS seeded from every origin at once: BFS always reaches a
    node for the first time via a shortest path regardless of how many downstream
    edges any node has, so nothing needs revisiting once discovered.
    """
    origins = find_origins(graph)
    depth: dict[str, int] = {origin: 0 for origin in origins}
    queue = collections.deque(origins)
    while queue:
        unit = queue.popleft()
        for downstream in graph.get(unit, []):
            if downstream in depth:
                continue
            depth[downstream] = depth[unit] + 1
            queue.append(downstream)
    return depth


def trace_paths(origin: str, graph: dict[str, list[str]]) -> list[list[str]]:
    """All downstream chains from one origin to an outlet, branching wherever a
    unit has more than one downstream edge. Not used by the WRD computation itself
    (which needs every unit's full ancestor set, not a single path) — kept for
    print_flow_chains and any other exploratory/debugging use."""

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
    """Diagnostic: print every origin's downstream chain(s) to its outlet(s). Not
    part of the WRD pipeline — a standalone way to eyeball the network topology,
    e.g. before trusting a hand-picked upstream/downstream huc4 split elsewhere
    (such as main.py's BASIN_MAPPINGS, or the 0807/0808/0809 mixup this module's
    ancestor-based approach was built to avoid)."""
    unit_destinations = load_unit_destinations(load_hu12s(huc2s, ignore_geometry=True), level)
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


def load_units(huc2s: list[str], level: int, hu12s: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Unit polygons for `level`, with their code in a "unit" column: huc4/huc8 from
    input/HUC's WBDHU{level} shapefiles (as main.py loads huc4s), huc12 from the
    NHDPlus GDB that the routing comes from."""
    if level == 12:
        units = hu12s[["huc12", "geometry"]].rename(columns={"huc12": "unit"})
    else:
        units = pandas.concat(
            [geopandas.read_file(path.join("input", "HUC", huc2, "Shape", f"WBDHU{level}.shp")) for huc2 in huc2s]
        )
        units.columns = [column.lower() for column in units.columns]
        units = units[[f"huc{level}", "geometry"]].rename(columns={f"huc{level}": "unit"})
    return geopandas.GeoDataFrame(units, geometry="geometry").reset_index(drop=True)


def rasterize_units(units: geopandas.GeoDataFrame, profile: dict[str, Any]) -> numpy.ndarray:
    """Label grid on `profile`'s grid: cell = 1 + its unit's row in `units`, 0 where
    no unit. all_touched=False (cell-centre rule) so a unit and its ancestors never
    share boundary cells; see utilities.clip_raster."""
    units_in_crs = units.to_crs(profile["crs"])
    return features.rasterize(
        zip(units_in_crs.geometry, range(1, len(units_in_crs) + 1)),
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        all_touched=False,
        dtype="int32",
    )


# ── Per-unit WRD ──────────────────────────────────────────────────────────────


def read_aligned(raster_path: str, profile: dict[str, Any] | None) -> tuple:
    """`raster_path` as float64 with nodata as NaN, checked to be on `profile`'s grid."""
    data, nodata, raster_profile = utilities.read_raster(raster_path)
    if profile is not None and (
        raster_profile["transform"] != profile["transform"] or data.shape != (profile["height"], profile["width"])
    ):
        raise ValueError(f"{raster_path} isn't on the same grid as the runoff raster")
    data = data.astype("float64")
    return numpy.where(utilities.valid_mask(data, nodata), data, numpy.nan), raster_profile


def compute_unit_wrd(
    labels: numpy.ndarray,
    unit_codes: list[str],
    ancestor_map: dict[str, set[str]],
    runoff: numpy.ndarray,
    consumption: numpy.ndarray,
    withdrawal: numpy.ndarray,
) -> tuple[numpy.ndarray, pandas.DataFrame]:
    """WRD for every unit with ancestors, each against its own ancestor set, in one
    pass over the label grid: per-unit sums come from a bincount, and each unit's
    Σb_M is the sum of its ancestors' (runoff - consumption) — the same as clipping
    the raster to the ancestors' union, since cell-centre labels never overlap.

    Returns the basin-wide WRD grid (NaN outside units with ancestors) and one row
    per such unit (code, n_ancestors, upstream_surplus, n_cells)."""
    n_labels = len(unit_codes) + 1
    flat_labels = labels.ravel()

    # PWS rasters are NaN wherever there's no public supply: that's zero withdrawal,
    # not missing data. Without this, b_L (and N_L) would only cover the ~5-20% of
    # cells that have public supply.
    withdrawal = numpy.where(~numpy.isnan(runoff) & numpy.isnan(withdrawal), 0.0, withdrawal)
    b_L = runoff - withdrawal

    unit_surplus = numpy.bincount(
        flat_labels, weights=numpy.nan_to_num(runoff).ravel(), minlength=n_labels
    ) - numpy.bincount(flat_labels, weights=numpy.nan_to_num(consumption).ravel(), minlength=n_labels)
    n_cells = numpy.bincount(flat_labels, weights=(~numpy.isnan(b_L.ravel())).astype(float), minlength=n_labels).astype(
        int
    )
    n_cells[0] = 0  # label 0 is "no unit"

    label_of = {code: label for label, code in enumerate(unit_codes, start=1)}
    # Per label: the divisor Σb_M / N_L, or NaN where the unit is skipped. Σb_M <= 0
    # gives WRD = 0 inside the unit (marked with an infinite divisor).
    divisor = numpy.full(n_labels, numpy.nan)
    rows = []
    for label, code in enumerate(unit_codes, start=1):
        ancestor_labels = [label_of[a] for a in ancestor_map.get(code, ()) if a in label_of]
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
    # b_L / inf is ±0.0 already; keep the "= 0" class from showing up as -0.0.
    wrd = numpy.where(numpy.isinf(divisor[labels]) & ~numpy.isnan(b_L), 0.0, wrd)
    wrd[~numpy.isfinite(wrd)] = numpy.nan

    return wrd, pandas.DataFrame(rows, columns=["unit", "n_ancestors", "upstream_surplus", "n_cells"])


def unit_stats(wrd: numpy.ndarray, labels: numpy.ndarray, unit_codes: list[str]) -> pandas.DataFrame:
    """utilities.compute_stats's min/q1/median/mean/q3/max for every unit at once."""
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
    output_path: str,
    title: str,
) -> None:
    """Per-cell WRD raster, classified into the same discrete bins as the notebook's
    WRD legend (colors reversed to complement WRC) — shows individual raster cells
    rather than plot_network_wrd's one-flat-color-per-unit choropleth. No basemap
    fetch, to keep this dependency-free for batch per-tier runs; the huc4
    boundaries are drawn instead for spatial context."""
    eps = 1e-6
    boundaries = [-100, -2, -1, -eps, eps, 1, 2, 100]
    colors = [
        "blue",
        "deepskyblue",
        "lightblue",
        "lightgray",
        "yellow",
        "orange",
        "darkred",
    ]

    counts = {
        "≤ -2": int(numpy.nansum(wrd <= -2)),
        "-2 to -1": int(numpy.nansum((wrd > -2) & (wrd <= -1))),
        "-1 to 0": int(numpy.nansum((wrd > -1) & (wrd < -eps))),
        "≈ 0": int(numpy.nansum((wrd >= -eps) & (wrd <= eps))),
        "0 to 1": int(numpy.nansum((wrd > eps) & (wrd <= 1))),
        "1 to 2": int(numpy.nansum((wrd > 1) & (wrd <= 2))),
        "≥ 2": int(numpy.nansum(wrd > 2)),
    }
    print(f"  WRD category counts: {counts}")

    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad(alpha=0)  # transparent nodata
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=7)

    height, width = wrd.shape
    left, bottom, right, top = rasterio.transform.array_bounds(height, width, profile["transform"])
    hucs_proj = hucs.to_crs(profile["crs"])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("white")
    img = ax.imshow(
        wrd.astype("float32"),
        extent=[left, right, bottom, top],
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )
    hucs_proj.boundary.plot(ax=ax, color="black", linewidth=0.4, zorder=3, aspect=None)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal", adjustable="datalim")

    tick_positions = [
        (-100 + -2) / 2,
        (-2 + -1) / 2,
        (-1 + -eps) / 2,
        0.0,
        (eps + 1) / 2,
        (1 + 2) / 2,
        (2 + 100) / 2,
    ]
    tick_labels = ["≤ −2", "−2 to −1", "−1 to 0", "= 0", "0 to 1", "1 to 2", "≥ 2"]
    cbar = fig.colorbar(img, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("WRD Categories")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  WRD cell-level plot saved → {output_path}")


def plot_network_wrd(
    units: geopandas.GeoDataFrame,
    unit_summary: pandas.DataFrame,
    level: int,
    output_path: str,
    title: str,
) -> None:
    """One basin-wide choropleth: each unit colored by its own mean WRD (computed
    against its own true ancestor set — not blended with any other unit)."""
    plot_df = units.copy()
    plot_df["mean_wrd"] = plot_df["unit"].map(unit_summary.set_index("unit")["mean"])
    # Thousands of huc12 outlines would drown out the fill.
    linewidth = 0.3 if level == 4 else 0.05

    eps = 1e-6
    boundaries = [-100, -2, -1, -eps, eps, 1, 2, 100]
    colors = [
        "blue",
        "deepskyblue",
        "lightblue",
        "lightgray",
        "yellow",
        "orange",
        "darkred",
    ]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    has_value = plot_df["mean_wrd"].notna()
    plot_df[~has_value].plot(ax=ax, color="whitesmoke", edgecolor="black", linewidth=linewidth)
    plot_df[has_value].plot(
        column="mean_wrd",
        cmap=cmap,
        norm=norm,
        ax=ax,
        edgecolor="black",
        linewidth=linewidth,
    )

    tick_positions = [
        (-100 + -2) / 2,
        (-2 + -1) / 2,
        (-1 + -eps) / 2,
        0.0,
        (eps + 1) / 2,
        (1 + 2) / 2,
        (2 + 100) / 2,
    ]
    tick_labels = ["≤ −2", "−2 to −1", "−1 to 0", "= 0", "0 to 1", "1 to 2", "≥ 2"]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label(f"Mean WRD by huc{level}")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Network WRD map saved → {output_path}")


def build_level(huc2s: list[str], level: int, hu12s: geopandas.GeoDataFrame) -> dict[str, Any]:
    """Everything about `level`'s network that doesn't change between scenarios and
    years: unit polygons, true ancestor sets, hop tiers and outlets."""
    graph = build_flow_graph(load_unit_destinations(hu12s, level), level)

    outlets = [unit for unit, downstream in graph.items() if not downstream]
    print(f"huc{level}: {len(graph)} units, {len(outlets)} outlet(s)")

    # ancestor_map is the source of truth for Σb_M — every unit that ACTUALLY,
    # transitively drains into a given unit, found by walking the real graph. depth
    # (hop count from the nearest origin) is used only to decide which tier-N.tif a
    # unit's already-correct result gets bundled into — never for what counts as its
    # ancestor. The two disagree surprisingly often: 22 of 84 huc4 edges in the MRB
    # have a successor whose hop is NOT exactly one more than its predecessor's
    # (tributaries of very different lengths converge on shared trunk segments), so
    # "everything at a shallower hop" would silently miss real ancestors, or worse,
    # treat a genuine downstream unit as an ancestor.
    ancestor_map = build_ancestor_map(graph)
    depth = compute_depth_tiers(graph)
    max_hop = max(depth.values(), default=0)
    print(f"huc{level}: {max_hop} hop level(s) downstream of {len(find_origins(graph))} origin(s)")

    return {
        "level": level,
        "units": load_units(huc2s, level, hu12s),
        "ancestor_map": ancestor_map,
        "depth": depth,
        "max_hop": max_hop,
        "outlets": outlets,
    }


def run(
    huc2s: list[str],
    hucs: geopandas.GeoDataFrame,
    climate_models: list[str],
    years: list[int],
    root_output_directory: str,
    levels: list[int] = list(LEVELS),
) -> dict[int, dict[str, dict[int, dict]]]:
    """WRD at every huc level in `levels`, for every climate model (scenario) in
    `climate_models` and every year in `years`. The flow networks don't change
    between them, so they're built once and shared by every run; runoff is that
    scenario/year's GCM-averaged quickflow. `hucs` is the basin's huc4s."""
    hu12s = load_hu12s(huc2s)
    networks = [build_level(huc2s, level, hu12s) for level in levels]

    results: dict[int, dict[str, dict[int, dict]]] = {level: {} for level in levels}
    for climate_model in climate_models:
        for year in years:
            year_results = run_year(hucs, climate_model, year, root_output_directory, networks)
            for level, result in year_results.items():
                results[level].setdefault(climate_model, {})[year] = result
    return results


def prepare_basin_rasters(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: str,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Basin-wide runoff, consumption (irr + pws-cu + thermoelectric) and pws-sw
    withdrawal in mm/yr, NaN as nodata, all on the runoff raster's 1 km grid."""
    _, _, runoff_path = raster_preparation.prepare_runoff_raster_basin(hucs, climate_model, year, root_output_directory)
    _, _, consumption_path = raster_preparation.prepare_wrd_consumption_raster(hucs, root_output_directory, year)
    _, _, withdrawal_path = raster_preparation.prepare_withdrawal_raster_basin(hucs, root_output_directory, year)

    runoff, profile = read_aligned(runoff_path, None)
    consumption, _ = read_aligned(consumption_path, profile)
    withdrawal, _ = read_aligned(withdrawal_path, profile)
    return runoff, consumption, withdrawal, profile


def run_year(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: str,
    networks: list[dict[str, Any]],
) -> dict[int, dict]:
    runoff, consumption, withdrawal, profile = prepare_basin_rasters(hucs, climate_model, year, root_output_directory)

    return {
        network["level"]: run_level(
            hucs,
            climate_model,
            year,
            root_output_directory,
            network,
            runoff,
            consumption,
            withdrawal,
            profile,
        )
        for network in networks
    }


def run_level(
    hucs: geopandas.GeoDataFrame,
    climate_model: str,
    year: int,
    root_output_directory: str,
    network: dict[str, Any],
    runoff: numpy.ndarray,
    consumption: numpy.ndarray,
    withdrawal: numpy.ndarray,
    profile: dict[str, Any],
) -> dict:
    level = network["level"]
    units = network["units"]
    depth = network["depth"]
    max_hop = network["max_hop"]
    unit_codes = units["unit"].tolist()

    out_dir = path.join(root_output_directory, "wrd", f"huc{level}", climate_model, str(year))
    os.makedirs(out_dir, exist_ok=True)

    # Pass 1: WRD per individual unit, each against its own true ancestor set
    # (ancestor_map) — this is the part that must be correct.
    labels = rasterize_units(units, profile)
    wrd, unit_summary = compute_unit_wrd(labels, unit_codes, network["ancestor_map"], runoff, consumption, withdrawal)
    unit_summary["hop"] = unit_summary["unit"].map(depth)
    unit_summary = unit_summary.merge(unit_stats(wrd, labels, unit_codes), on="unit", how="left")
    print(
        f"huc{level} ({climate_model}, {year}): WRD for {len(unit_summary)} of "
        f"{len(units)} units (the rest are headwaters or have no valid cells)"
    )
    if level == 4:
        for row in unit_summary.itertuples():
            print(
                f"{row.unit} (hop {row.hop}): {row.n_ancestors} true upstream huc4s, "
                f"Σb_M={row.upstream_surplus:.2f}, N_L={row.n_cells}"
            )

    # Pass 2: snapshot the composite at each tier — tier-N's file covers every unit
    # from hop 1 through N, each cell carrying the value computed for its own unit's
    # true ancestry above. Finer levels write only the last (full) tier.
    hop_of_label = numpy.full(len(unit_codes) + 1, numpy.nan)
    hop_of_label[1:] = [depth.get(code, numpy.nan) for code in unit_codes]
    cell_hop = hop_of_label[labels]
    snapshot_hops = range(1, max_hop + 1) if level == 4 else [max_hop]

    results = {}
    for hop in snapshot_hops:
        cumulative_wrd = numpy.where(cell_hop <= hop, wrd, numpy.nan)
        cumulative_units = unit_summary.loc[unit_summary["hop"] <= hop, "unit"].tolist()
        if not cumulative_units:
            continue

        utilities.save_raster(cumulative_wrd, profile, path.join(out_dir, f"tier-{hop}.tif"))
        stats = utilities.compute_stats(
            cumulative_wrd,
            f"WRD huc{level} tier {hop} (hops 1-{hop}, {climate_model}, {year})",
        )
        # TODO: restore once plot_histogram lives in utilities.py
        # utilities.plot_histogram(
        #     cumulative_wrd,
        #     path.join(out_dir, f"tier-{hop}-hist.png"),
        #     f"WRD tier {hop} (hops 1-{hop}, {climate_model}, {year}) Distribution",
        #     "WRD",
        # )
        plot_wrd_cells(
            cumulative_wrd,
            profile,
            hucs,
            path.join(out_dir, f"tier-{hop}-cells.png"),
            f"WRD huc{level} tier {hop} (hops 1-{hop}, {climate_model}, {year})",
        )

        results[hop] = {
            "units": cumulative_units,
            "includes_outlet": any(u in network["outlets"] for u in cumulative_units),
            "stats": stats,
        }

    summary_path = path.join(out_dir, f"huc{level}_summary.csv")
    unit_summary.rename(columns={"unit": f"huc{level}"})[
        [f"huc{level}", "hop", "n_ancestors", "upstream_surplus", "n_cells", "min", "q1", "median", "mean", "q3", "max"]
    ].sort_values(f"huc{level}").to_csv(summary_path, index=False)
    print(f"Per-huc{level} summary saved → {summary_path}")

    tier_summary = pandas.DataFrame(
        [
            {
                "tier": tier,
                f"huc{level}s": ",".join(sorted(r["units"])),
                f"n_huc{level}s": len(r["units"]),
                "includes_outlet": r["includes_outlet"],
                **r["stats"],
            }
            for tier, r in results.items()
        ]
    ).sort_values("tier")
    tier_summary_path = path.join(out_dir, "tier_summary.csv")
    tier_summary.to_csv(tier_summary_path, index=False)
    print(f"Cumulative tier summary saved → {tier_summary_path}")

    plot_network_wrd(
        units,
        unit_summary,
        level,
        path.join(out_dir, "network-wrd.png"),
        f"WRD by huc{level} — {climate_model} {year}",
    )

    return results

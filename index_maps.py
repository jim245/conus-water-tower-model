"""
Plots the class maps of RWY and WRC, for the basin maps (main.py), the network maps (network_indices.py) and the
CONUS maps (conus_maps.py).
"""

import itertools
from pathlib import Path
from typing import Any

import geopandas
import matplotlib.colors as mcolors
import numpy

import utilities

RWY_BOUNDARIES = [0, 0.5, 1, 2, 3, 5, 10]
RWY_COLORS = [
    "lightgray",  # = 0
    "#f5d300",  # 0 to 0.5: yellow
    "#fff3a0",  # 0.5 to 1: lighter yellow
    "#f9b4c8",  # 1 to 2: pink
    "#e7609a",  # 2 to 3: dark pink
    "#e31a1c",  # 3 to 5: red
    "#b10026",  # 5 to 10: darker red
    "#67000d",  # > 10: dark red
]
RWY_LABELS = [
    "0",
    "0 to 0.5",
    "0.5 to 1",
    "1 to 2",
    "2 to 3",
    "3 to 5",
    "5 to 10",
    "> 10",
]

WRC_COLORS = ["orange", "yellow", "green", "lightblue"]
WRC_LABELS = ["≤ -1", "-1 to 0", "0 to 1", "> 1"]


def plot_rwy_classes(
    rwy: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Classifies the RWY into the classes of RWY_LABELS, prints the class counts and plots the class map.

    The classes are upper-inclusive (0 < RWY <= 0.5, ...). A cell with RWY <= 0 has a class of its own (no
    quickflow).

    Parameters:
        rwy (numpy.ndarray): The RWY.
        profile (dict[str, Any]): The profile of the RWY grid.
        hucs (geopandas.GeoDataFrame): The HUCs whose outlines to plot.
        output_path (Path): The path to the output PNG.
        title (str): The map title.
    """

    rwy_classes = numpy.full(rwy.shape, numpy.nan, dtype="float32")
    rwy_classes[rwy <= 0] = 0
    for i, (low, high) in enumerate(itertools.pairwise(RWY_BOUNDARIES)):
        rwy_classes[(rwy > low) & (rwy <= high)] = i + 1
    rwy_classes[rwy > RWY_BOUNDARIES[-1]] = len(RWY_BOUNDARIES)

    counts = {label: int(numpy.sum(rwy_classes == i)) for i, label in enumerate(RWY_LABELS)}
    print(f"  RWY category counts: {counts}")

    n_classes = len(RWY_COLORS)
    utilities.plot_raster_map(
        rwy_classes,
        profile,
        hucs,
        output_path,
        title=title,
        cmap=mcolors.ListedColormap(RWY_COLORS),
        norm=mcolors.BoundaryNorm(boundaries=numpy.arange(n_classes + 1) - 0.5, ncolors=n_classes),
        cbar_label="Runoff Water Yield (RWY)",
        cbar_ticks=list(range(n_classes)),
        cbar_ticklabels=RWY_LABELS,
    )


def plot_wrc_classes(
    wrc: numpy.ndarray,
    profile: dict[str, Any],
    hucs: geopandas.GeoDataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Classifies the WRC into the classes of WRC_LABELS, prints the class counts and plots the class map.

    The classes are the WRC classes of the original method: WRC <= -1, -1 < WRC < 0, 0 <= WRC <= 1 and WRC > 1.

    Parameters:
        wrc (numpy.ndarray): The WRC.
        profile (dict[str, Any]): The profile of the WRC grid.
        hucs (geopandas.GeoDataFrame): The HUCs whose outlines to plot.
        output_path (Path): The path to the output PNG.
        title (str): The map title.
    """

    wrc_classes = numpy.full(wrc.shape, numpy.nan, dtype="float32")
    wrc_classes[wrc <= -1] = 0
    wrc_classes[(wrc > -1) & (wrc < 0)] = 1
    wrc_classes[(wrc >= 0) & (wrc <= 1)] = 2
    wrc_classes[wrc > 1] = 3

    counts = {label: int(numpy.sum(wrc_classes == i)) for i, label in enumerate(WRC_LABELS)}
    print(f"  WRC category counts: {counts}")

    n_classes = len(WRC_COLORS)
    utilities.plot_raster_map(
        wrc_classes,
        profile,
        hucs,
        output_path,
        title=title,
        cmap=mcolors.ListedColormap(WRC_COLORS),
        norm=mcolors.BoundaryNorm(boundaries=numpy.arange(n_classes + 1) - 0.5, ncolors=n_classes),
        cbar_label="Water Resource Contribution (WRC)",
        cbar_ticks=list(range(n_classes)),
        cbar_ticklabels=WRC_LABELS,
    )

# Overview

This repository computes three water-tower indices for the basins of the conterminous United States (CONUS), in a single pipeline driven by [main.py](./main.py). The Relative Water Yield (RWY) tells how much runoff a cell makes, compared with the runoff downstream. The Water Resource Contribution (WRC) tells how much a cell supplies to the water use downstream. The Water Resource Dependency (WRD) tells how much a hydrologic unit depends on the water of its upstream units. The pipeline uses the Quickflow runoff of two climate scenarios (SSP245 and SSP370) and the gridded water use of 2015 to 2020. It gives each index at the basin level and at three network levels of the Hydrologic Unit Code (HUC) watersheds: HUC-4, HUC-8 and HUC-12. The outputs are basin maps and CONUS maps.

# Setup

Use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for package and environment management. The environment requires linux-64.

From the repository root, run `conda env create -f environment.yaml -p .conda` to create the environment in `.conda`, and `conda activate ./.conda` to activate it. Without activation, `PROJ_DATA` is not set, and the geospatial libraries can't find the PROJ database.

Run `.conda/bin/pre-commit install` once to install the [pre-commit hooks](./.pre-commit-config.yaml). They lint, type-check, spell-check and format each commit, according to the tool settings in [`pyproject.toml`](./pyproject.toml).

## Folder Structure

All inputs are in `input/`. Put the datasets at the paths in [Datasets Used](#datasets-used) before the run.

The pipeline writes its outputs to `output/`. All rasters are single-band float32 GeoTIFFs with NaN as no-data. `{ssp}` is the scenario, `{level}` is 4, 8 or 12, and a `diff` file is the last year minus the first year.

- `output/national/`: the national rasters that the basin stages clip. These are the mean Quickflow of each scenario and year, its 1 km reprojection in mm/yr, and the water use in mm/yr.
- `output/{basin}/rwy/{ssp}/` and `output/{basin}/wrc/{ssp}/`: the basin-level RWY and WRC, `{ssp}_{year}_{rwy,wrc}.tif`, `{ssp}_diff_{rwy,wrc}.tif`, and a `.png` map of each.
- `output/{basin}/{rwy,wrc}/huc{level}/{ssp}/`: the network RWY and WRC, `{ssp}_{year}_{rwy,wrc}.tif` and `.png`.
- `output/{basin}/wrd/huc{level}/{ssp}/`: the WRD. `{ssp}_{year}_wrd.tif` and `.png` give the WRD of each cell, and `{ssp}_{year}_network.png` the mean WRD of each unit. `{ssp}_{year}_summary.csv` has one row for each unit with a WRD: the columns `huc{level}`, `n_ancestors`, `upstream_surplus` (Σb_M), `n_cells` (N_L), and the `min`, `q1`, `median`, `mean`, `q3` and `max` of the unit's WRD.
- `output/{basin}/` also holds the clips of the inputs (`quickflow/`, `runoff/`, `irr/`, `pws/`, `pws-wd/`, `thermoelectric/`, `{year}-consumption_total.tif`, `{year}-wrd-consumption-basin.tif`).
- `output/CONUS/`: the CONUS mosaic of each basin-level and network-level product above, in the same folder layout, with a diff of each.

# Workflow

1. Prepare the [national rasters](./raster_preparation.py) once for all basins. These are the cell-by-cell mean Quickflow over the GCMs of each scenario and its reprojection onto the 1 km water-use grid. The water use is also converted to mm/yr there.
2. For each basin of `CONUS_BASINS` in [main.py](./main.py), compute the basin-level RWY and WRC. They compare each upstream cell of the basin with one downstream outlet region (see [Basins](#basins)).
3. Build the flow network of each level from the HUC-12 routing. Compute the [WRD](./wrd_flows.py) and the network [RWY and WRC](./network_indices.py) of each scenario and year. The network indices compare each cell with all the units downstream of its unit.
4. [Mosaic](./conus_maps.py) the basin outputs into CONUS maps, and draw the [class maps](./index_maps.py).

Run all commands from the repository root:

```sh
python main.py
python conus_maps.py
```

`main.py` computes the basins one after the other. The largest basin (MRB) needs about 53 minutes and 9.7 GB of memory, so don't run basins in parallel on a machine with 16 GB. The national rasters need about 6 minutes, and the CONUS map outlines about 18 minutes.

# Method

## Symbols

| Symbol | Meaning                                                                      | Grid                  |
| ------ | ---------------------------------------------------------------------------- | --------------------- |
| q      | Quickflow, the mean over the GCMs of the scenario                            | 800 m, Quickflow grid |
| Q, qi  | The same Quickflow, in mm/yr                                                 | 1 km, water-use grid  |
| WW     | Surface-water public-supply withdrawal (pws-sw), in mm/yr                    | 1 km                  |
| wc     | Consumption: irrigation, public supply (pws-cu) and thermoelectric, in mm/yr | 1 km                  |

## RWY and WRC

At the basin level, the indices compare each upstream cell with the downstream outlet region D of the basin:

```text
RWY_cell = q_cell / mean(q over D)
WRC_cell = (mean(Q over D) - mean(WW over D)) / (qi_cell - wc_cell)
```

At the network levels, D is the set of all the units downstream of the cell's unit in the flow network. A mean over D is the mean of all its cells, not a mean of the unit means. A unit without downstream units (an outlet or a closed-basin sink) has no RWY and no WRC.

- A mean uses only the finite, non-negative cells.
- Inside the runoff area, a water-use cell with no value counts as 0. pws-sw has no value outside the water service areas, where there is no public supply.
- A cell with |qi - wc| less than 1e-6 mm/yr has no WRC.
- The RWY has no value when the mean downstream Quickflow is not positive.

The WRC of the original method multiplies the downstream withdrawal by 0.75. That factor was for the Potomac study only, and this pipeline does not use it.

## WRD

Each unit L of a level gets:

```text
WRD_L = b_L / (Σb_M / N_L)

b_L  = Q - WW             for each cell of L
Σb_M = Σqi_A - Σwc_A      summed over all cells of all ancestors A of L
N_L  = number of cells of L with a value of b_L
```

A unit without ancestors (a headwater) has no WRD. The WRD is 0 in a unit where Σb_M is not positive. Σb_M does not subtract the withdrawals of the units between an ancestor and L. Thus Σb_M increases downstream, and the terminal unit gets the surplus of the full basin. This is intentional.

## Flow Network

The network comes from the `tohuc` routing of the HUC-12s in the NHDPlus HR geodatabase. At HUC-4 and HUC-8, unit A has an edge to unit B if a HUC-12 in A routes to a HUC-12 in B. The unit polygons of all levels are the HUC-12s of the geodatabase, dissolved by code prefix. Thus the polygons and the network have the same codes. Two hand-checked tables in [wrd_flows.py](./wrd_flows.py) correct the routing:

- `TOHUC_REPLACEMENTS` and `TOHUC_OVERRIDES` give the downstream HUC-12 of the `tohuc` codes that are in no HUC-12 of the geodatabase (the Great Lakes and four inland rivers).
- `ARTIFACT_EDGES` removes the HUC-4 and HUC-8 edges that are not real flow between the units. Each of them made a cycle, or made a full basin an ancestor of its neighbour.

## Classes

The `.png` maps show these classes. The GeoTIFFs hold the values.

| Index | Classes                                                                              |
| ----- | ------------------------------------------------------------------------------------ |
| RWY   | 0, 0 to 0.5, 0.5 to 1, 1 to 2, 2 to 3, 3 to 5, 5 to 10, > 10 (upper bound included) |
| WRC   | ≤ -1, -1 to 0, 0 to 1, > 1                                                           |
| WRD   | < -2, -2 to -1, -1 to 0, = 0, 0 to 1, 1 to 2, ≥ 2                                    |

# Basins

The basins of the CONUS maps do not overlap. The Mississippi and the Colorado are whole river systems (MRB, CRB), not their HUC-2 regions. HUC-2 09 (Souris-Red-Rainy) is in no basin.

| Basin                       | HUC-2                  | Downstream outlet region (HUC-4)   |
| --------------------------- | ---------------------- | ---------------------------------- |
| NER New England             | 01                     | 0101, 0105, 0106, 0108, 0109, 0110 |
| MAR Mid-Atlantic            | 02                     | 0203, 0204, 0208                   |
| SAG South Atlantic-Gulf     | 03                     | 0301-0308, 0310-0314, 0317, 0318   |
| GLR Great Lakes             | 04                     | 0429, 0430, 0431                   |
| MRB Mississippi River Basin | 05, 06, 07, 08, 10, 11 | 0809                               |
| TGR Texas-Gulf              | 12                     | 1204, 1209, 1210, 1211             |
| RGR Rio Grande              | 13                     | 1305, 1309                         |
| CRB Colorado River Basin    | 14, 15                 | 1503                               |
| GBR Great Basin             | 16                     | 1602, 1603, 1605, 1606             |
| PNW Pacific Northwest       | 17                     | 1708, 1710, 1711, 1712             |
| CAR California              | 18                     | 1801, 1803, 1805-1810              |

The upstream area of a basin is the basin without its outlet region. Some HUC-4s are not in the flow network of their basin (0807 and 0808 in the MRB, 1508 in the CRB). They are also not in the upstream area.

Blank areas in the maps:

- At the basin level, the RWY and the WRC are for the upstream cells only, so the outlet region is blank.
- At the network levels, a unit without downstream units has no RWY and no WRC, and a headwater has no WRD. At HUC-4 this makes large blank areas on the coasts.
- Quickflow has no cells in Canada or Mexico.

# Known Limitations

- Inter-basin transfers are not in the model. The consumption is where people use the water, not where the water leaves the river. A unit that imports water looks like it consumes its own supply (for example the Gila HUC-4s, through the Central Arizona Project). An export out of the basin is not a depletion.
- A unit that drains to more than one downstream unit adds its full surplus to each of them. The model does not divide a diverging flow between its branches.
- Inflow from outside the US is not counted. A US unit downstream of Canada or Mexico (for example the St. Lawrence, the Rio Grande below the Rio Conchos, the Columbia) looks more dependent than it is.
- The consumption is surface water, with one exception: pws-cu is the public-supply consumption from all sources, so it includes the consumption from groundwater. The source split of the irrigation and thermoelectric consumption is not known.
- pws-cu and pws-tot come from independent disaggregations. Some cells have a consumption larger than the total withdrawal (small).
- A Quickflow cell gets the mean of the GCMs that have a value there. In SSP370, BCCCSM2MR has a different valid mask from the other four GCMs. Thus some cells on the land and water edge use 1 or 4 GCMs.
- The HUC-12s of the geodatabase are not a valid coverage: there are small gaps and overlaps between them. The map outlines don't show the gaps smaller than one 1 km cell.

# Datasets Used

Paths are relative to `input/`. Quickflow has no unit metadata, and the pipeline assumes m/yr. The water-use rasters are in annual mean million gallons per day (mgd) per cell, and the pipeline converts them to mm/yr with the cell area.

## Hydrography

| Dataset                                                                                                          | Path                                                    | Use                            |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------ |
| [NHDPlus High Resolution](https://www.usgs.gov/national-hydrography/nhdplus-high-resolution), National Release 2 | `NHDPlus_H_National_Release_2_GDB.gdb`, layer `WBDHU12` | Unit polygons, `tohuc` routing |

## Runoff

| Dataset                              | Path                                           | Use                                                                                                                     |
| ------------------------------------ | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Quickflow (TODO: source and version) | `Quickflow_Paige/QF_{GCM}_{SSP}/Qf_{year}.tif` | RWY, and the runoff of WRC and WRD. GCMs: BCCCSM2MR, FGOALSG3, GFDLESM4, INMCM48, MIROC6. Some file names start with `QF_` |

## Water Use

The water-use rasters are on the 1000 m EPSG:5070 grid. They come from two related pipelines: [huc-water-use-disaggregation](https://github.com/jim245/huc-water-use-disaggregation) (irr, pws) and [thermoelectric-spatial-disaggregation](https://github.com/jim245/thermoelectric-spatial-disaggregation).

| Dataset                                   | Path                                           | Use                |
| ----------------------------------------- | ---------------------------------------------- | ------------------ |
| Irrigation consumption                    | `irr/irr_huc12_{year}.tif`                     | wc                 |
| Public-supply consumption, all sources    | `pws-cu/pws-cu_huc12_{year}.tif`               | wc                 |
| Public-supply surface-water withdrawal    | `pws-sw/pws-sw_huc12_{year}.tif`               | WW                 |
| Public-supply total withdrawal            | `pws-tot/pws-tot_huc12_{year}.tif`             | The 1 km grid only |
| Thermoelectric consumption                | `thermoelectric/{year}-annual-consumption.tif` | wc                 |

# Citation

TODO: the papers of the method.

# License

Pending

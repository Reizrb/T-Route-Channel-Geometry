# T-Route Channel Geometry

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23520058.svg)](https://doi.org/10.5281/zenodo.23520058)

Python code to update the channel geometry that
[T-Route](https://github.com/NOAA-OWP/t-route), the routing model of the National Water
Model (NWM), uses for every river reach.

T-Route's Muskingum-Cunge routing treats each reach as a trapezoidal channel, defined in
the NWM **RouteLink** file by its bottom width, top width, and side slope. This code
replaces that geometry with machine-learning predictions of bankfull width and depth from
[HydroGeom](https://github.com/Reizrb/HydroGeom) (Zarrabi et al., 2025), for a channel
shape you choose, and recomputes the Muskingum routing parameters. The result is a new
RouteLink file that T-Route can use directly.

## Requirements

- Python 3.10 or later
- Internet access: the original NWM RouteLink file (`RouteLink_CONUS.nc`, several hundred
  MB) is downloaded from NOAA, and the channel geometry from the HydroGeom API (or use the
  full CONUS geometry dataset from [Zenodo](https://doi.org/10.5281/zenodo.19208847))

## Install

```bash
git clone https://github.com/Reizrb/TRoute-Channel-Geometry
cd TRoute-Channel-Geometry
pip install numpy pandas netCDF4 "git+https://github.com/Reizrb/HydroGeom" jupyter
```

## Run the example

```bash
jupyter notebook examples/TRoute_geometry_HUC2-11_a0.4.ipynb
```

The notebook updates the reaches of the **Arkansas-White-Red region (HUC2 11)** with a
trapezoidal channel of **a = 0.4**, writes the new RouteLink file, compares original and
updated values, and shows how to point T-Route to the new file. The original RouteLink
file is downloaded from NOAA automatically.

## Use it

```python
import troute_geometry as tg

tg.download_routelink("RouteLink_CONUS.nc")                  # original NWM file from NOAA
geometry = tg.channel_geometry(huc2="11")                    # or state="AR", huc8=..., comids=[...]
# geometry = tg.read_geometry_file("Bankfull_Meanflow_CONUS_Stream_Reaches.txt")   # all of CONUS

tg.update_routelink("RouteLink_CONUS.nc", "RouteLink_CONUS_a0.4.nc", geometry, a=0.4)
```

Then set the new file in the T-Route configuration YAML:

```yaml
network_topology_parameters:
  supernetwork_parameters:
    geo_file_path: RouteLink_CONUS_a0.4.nc
```

| Setting | Meaning |
|---------|---------|
| `a` | Bottom width / top width of the trapezoid: `0` triangle, between 0 and 1 trapezoid, `0.99` nearly rectangular |

| Function in `troute_geometry.py` | What it does |
|----------------------------------|--------------|
| `download_routelink(dest, url=None)` | Download the original CONUS RouteLink file from NOAA (skipped if it exists) |
| `channel_geometry(huc2=, state=, huc8=, comids=)` | Bankfull width and depth for an area from HydroGeom |
| `read_geometry_file(path)` | Bankfull width and depth from the full CONUS dataset file |
| `update_routelink(routelink_in, routelink_out, geometry, a)` | Write a new RouteLink file with the updated geometry |
| `compare(routelink_in, routelink_out, comids=None)` | Original vs updated values of the geometry and Muskingum variables |

## Ready-made RouteLink files

Updated RouteLink files for six channel shapes across CONUS (a = 0, 0.2, 0.4, 0.6, 0.8, and 0.99), made with this code, are on Zenodo: https://doi.org/10.5281/zenodo.23520462

## The original RouteLink file

`download_routelink()` gets `RouteLink_CONUS.nc` from NOAA's NWM production server
(`https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/`), from the **current** operational NWM
release (for example `nwm.v3.1.6`). NOAA keeps only the current release there, so the
version is looked up each time, and the file can change when NWM is updated. To use a
specific file, pass its URL (`url=...`) or download it yourself and set its path. The
original file is never modified.

## What it changes

For each reach with bankfull top width B and depth d, and bottom/top width ratio a:

| RouteLink variable | New value |
|--------------------|-----------|
| `TopWdth` | B |
| `BtmWdth` | a × B |
| `ChSlp` | 1 / m, where m = (1 − a) B / (2d) is the H:V side slope (T-Route stores V:H) |
| `TopWdthCC` | 3 × B |
| `MusK` | Δx / c<sub>k</sub>, the travel time through the reach (s) |
| `MusX` | 0.5 × [1 − Q / (c<sub>k</sub> B S<sub>0</sub> Δx)], clipped to 0–0.5 |

Q is the bankfull discharge from Manning's equation and c<sub>k</sub> = dQ/dA the kinematic
wave speed, both for the trapezoid at bankfull depth, with the reach's Manning's n, slope
S<sub>0</sub>, and length Δx from the RouteLink file. All other RouteLink variables are
unchanged.

Notes:

- `MusK`, `MusX`, and `TopWdthCC` are recomputed for **every** reach in the file, including
  reaches that keep their original geometry, so all reaches use the same formulas.
- `a = 1` (exactly rectangular) is not allowed: it gives a zero side slope that the T-Route
  Muskingum-Cunge kernel cannot represent. Use `a = 0.99` instead.
- `a = 0` (triangular) gives a zero bottom width; check routing stability on a few reaches.
- Reaches without a geometry prediction keep their original geometry.

## Citation

- Zarrabi, R., McDermott, R., Erfani, S. M. H., & Cohen, S. (2025). Bankfull and
  mean-flow channel geometry estimation through machine learning algorithms across
  the CONtiguous United States (CONUS). *Water Resources Research*, 61(2).
  https://doi.org/10.1029/2024WR037997

## Funding

<table><tr>
<td width="110"><img src="assets/ciroh_logo.png" alt="CIROH logo" width="100"></td>
<td>Funding for this project was provided by the National Oceanic &amp; Atmospheric Administration (NOAA), awarded to the Cooperative Institute for Research to Operations in Hydrology (CIROH) through the Cooperative Agreement with The University of Alabama (NA22NWS4320003).</td>
</tr></table>

## License

[MIT](LICENSE)

## Related repositories

| Repository | What it does |
|------------|--------------|
| [Bankfull-and-mean-flow-channel-geometry-for-CONUS](https://github.com/Reizrb/Bankfull-and-mean-flow-channel-geometry-for-CONUS) | ML models and the CONUS bankfull and mean-flow channel geometry dataset |
| [HydroGeom](https://github.com/Reizrb/HydroGeom) | Web page, API, and Python package to get and use the channel geometry |
| [Bathymetric-Adjustment-of-Rating-Curves-and-Stage-Shift-BARC-SS](https://github.com/Reizrb/Bathymetric-Adjustment-of-Rating-Curves-and-Stage-Shift-BARC-SS) | Adds the channel geometry to OWP HAND-FIM rating curves for flood mapping (BARC, BARC-SS) |
| [TRoute-Channel-Geometry](https://github.com/Reizrb/TRoute-Channel-Geometry) | Adds the channel geometry to the NWM RouteLink file for T-Route routing (this repository) |
| [T-Route](https://github.com/NOAA-OWP/t-route) | NOAA OWP's channel routing model for the National Water Model |

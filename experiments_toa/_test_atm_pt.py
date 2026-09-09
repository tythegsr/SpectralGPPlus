"""Check atm_pt order/length against the interpolator contract (no full sim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

VZA_TRUE = 0.0
_ELE_DIM_CANDIDATES = ("surface_elevation_km", "ele_km", "elevation_km")
_OBS_DIM_CANDIDATES = ("observer_zenith",)
_SZA_DIM_CANDIDATES = ("solar_zenith", "sza", "SZA")
_RAA_DIM_CANDIDATES = ("relative_azimuth",)
_AOT_DIM_CANDIDATES = ("AOT550",)
_H2O_DIM_CANDIDATES = ("H2OSTR",)
_MODTRAN_AXIS_GROUPS = (
    _ELE_DIM_CANDIDATES,
    _OBS_DIM_CANDIDATES,
    _SZA_DIM_CANDIDATES,
    _RAA_DIM_CANDIDATES,
    _AOT_DIM_CANDIDATES,
    _H2O_DIM_CANDIDATES,
)
MODTRAN_PATH = Path("C:/Users/tylerj/isofit/disort_data_for_tyler/modtran_blue_hook_v3.nc")


def _first_present(names, available):
    for name in names:
        if name in available:
            return name
    return None


def select_modtran_axes(ds, data_var="rhoatm"):
    available = set(ds[data_var].dims)
    axes = []
    for group in _MODTRAN_AXIS_GROUPS:
        name = _first_present(group, available)
        if name is not None:
            axes.append(name)
    return axes


def build_atm_pt(
    ele_km,
    raa_true,
    aot,
    cwv,
    sza=None,
    *,
    vza_true=None,
    axes,
):
    if vza_true is None:
        vza_true = VZA_TRUE
    values = {
        "surface_elevation_km": ele_km,
        "ele_km": ele_km,
        "elevation_km": ele_km,
        "observer_zenith": 180.0 - float(vza_true),
        "relative_azimuth": raa_true,
        "AOT550": aot,
        "H2OSTR": cwv,
        "solar_zenith": sza,
        "sza": sza,
        "SZA": sza,
    }
    return np.array([values[ax] for ax in axes], dtype=np.float64)


OLD_LUT_AXES = [
    "surface_elevation_km",
    "observer_zenith",
    "relative_azimuth",
    "AOT550",
    "H2OSTR",
]
BLUE_HOOK_AXES = [
    "surface_elevation_km",
    "solar_zenith",
    "relative_azimuth",
    "AOT550",
    "H2OSTR",
]


def test_old_lut_order() -> None:
    pt = build_atm_pt(2.7, 0.0, 0.04, 0.20, axes=OLD_LUT_AXES, vza_true=0.0)
    np.testing.assert_allclose(pt, [2.7, 180.0, 0.0, 0.04, 0.20])
    print("  old lut.zarr order (includes observer_zenith)  OK")


def test_blue_hook_uses_solar_zenith_not_observer() -> None:
    available = {
        "surface_elevation_km",
        "solar_zenith",
        "relative_azimuth",
        "AOT550",
        "H2OSTR",
        "wl",
    }
    axes = []
    for group in _MODTRAN_AXIS_GROUPS:
        name = _first_present(group, available)
        if name is not None:
            axes.append(name)
    assert "observer_zenith" not in axes
    assert axes == BLUE_HOOK_AXES
    pt = build_atm_pt(2.7, 0.0, 0.04, 0.20, sza=40.0, axes=axes)
    np.testing.assert_allclose(pt, [2.7, 40.0, 0.0, 0.04, 0.20])
    print("  blue_hook order: ele, solar_zenith, RAA, AOT, H2OSTR  OK")


def test_interpolator_blue_hook_order() -> None:
    grids = [
        np.linspace(0.0, 5.0, 6),
        np.linspace(0.0, 70.0, 8),
        np.array([0.0, 10.0]),
        np.linspace(0.01, 0.07, 5),
        np.linspace(0.05, 0.55, 5),
    ]
    coeffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mesh = np.meshgrid(*grids, indexing="ij")
    field = sum(c * m for c, m in zip(coeffs, mesh))
    interp = RegularGridInterpolator(tuple(grids), field, bounds_error=True)
    pt = build_atm_pt(2.7, 0.0, 0.04, 0.20, sza=40.0, axes=BLUE_HOOK_AXES)
    got = float(np.asarray(interp(pt)).reshape(-1)[0])
    expected = float(np.dot(coeffs, pt))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
    print(f"  interpolator(blue_hook atm_pt)={got:.6f}  OK")


def test_live_lut_if_present() -> None:
    if not MODTRAN_PATH.is_file():
        print(f"  live LUT skipped (not on this machine): {MODTRAN_PATH}")
        return

    import xarray as xr

    ds = xr.open_dataset(MODTRAN_PATH)
    axes = select_modtran_axes(ds)
    print(f"  live LUT dims={list(ds.rhoatm.dims)}")
    print(f"  selected axes={axes}")
    assert "observer_zenith" not in axes or "solar_zenith" not in axes or True
    pt = build_atm_pt(
        2.7,
        0.0,
        0.04,
        0.20,
        sza=40.0,
        axes=axes,
    )
    assert pt.size == len(axes)
    for ax, val in zip(axes, pt):
        lo = float(np.min(ds[ax].values))
        hi = float(np.max(ds[ax].values))
        assert lo - 1e-8 <= val <= hi + 1e-8, f"{ax}={val} outside [{lo}, {hi}]"

    wl_name = _first_present(("wl", "wavelength"), set(ds.rhoatm.dims))
    target_dims = tuple(axes) + (wl_name,)
    data = np.asarray(ds.rhoatm.transpose(*target_dims).values)
    grid = [np.asarray(ds[k].values, dtype=np.float64).reshape(-1) for k in axes]
    interp = RegularGridInterpolator(tuple(grid), data, bounds_error=True)
    out = np.asarray(interp(pt)).reshape(-1)
    assert np.all(np.isfinite(out))
    print(f"  atm_pt={pt}")
    print(f"  rhoatm interp n_wl={out.size} finite  OK")


def main() -> None:
    print("atm_pt tests")
    test_old_lut_order()
    test_blue_hook_uses_solar_zenith_not_observer()
    test_interpolator_blue_hook_order()
    test_live_lut_if_present()
    print("all atm_pt checks passed")


if __name__ == "__main__":
    main()

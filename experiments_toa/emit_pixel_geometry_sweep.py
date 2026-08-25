"""EMIT pixel geometry sweep: forward-model vs observed EMIT reflectance.

Standalone script (no repo / experiments_toa imports). Uses a **hard-coded** EMIT
pixel (15-D ISOFIT state + 285-band reflectance + wavelength grid) and the same LUT
paths as ``toa_simulations_august.py``. Sweeps cos_i and elevation; fixes VZA=6° and
MODTRAN LUT coszen. Compares simulated vs EMIT TOA reflectance.

Softmax convention (critical)
-----------------------------
EMIT stores z_snow, z_pv, z_npv, z_soil as **logits**. The forward model applies
``softmax([fsnow, fPV, fNPV, fsoil])`` internally, so pass those logits directly.
Do **not** pre-softmax before calling simulate_pixel — that would double-transform.

For reporting only, softmax is applied once to show physical cover fractions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from common import VectorInterpolator, calculate_resample_matrix

# Same paths as experiments_toa/toa_simulations_august.py
MODTRAN_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/lut.zarr"
DISORT_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/disort_snow_lut_EMIT.nc"
ENDMEMBER_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/endmembers.csv"
EMIT_WAVE_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/emit-wave.txt"
NOISE_PATH = "C:/Users/tylerj/isofit/disort_data_for_tyler/emit_noise.txt"
DEFAULT_OUT = "C:/Users/tylerj/isofit/disort_data_for_tyler/data/emit_pixel_geometry_sweep_onlyELE"

# ISOFIT snow / multi-surface state vector (15-D); sinA/cosA are aspect params, not cos_i.
EMIT_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "sinA",
    "cosA",
    "grain_radius",
    "liquid_water",
    "dust",
    "algae",
    "z_snow",
    "z_pv",
    "z_npv",
    "z_soil",
    "veg_rank",
    "npv_rank",
    "soil_rank",
    "AOT660",
    "H20STR",
)

RAA_TRUE = 164.0
VZA_DEFAULT = 6.0
RAA_DISORT = 180.0 - RAA_TRUE
ELE_DEFAULT = 3.0
COS_I_DEFAULT = 0.53
NIR_TARGET_NM = 900.0

EMIT_STATE_INDEX = {name: i for i, name in enumerate(EMIT_STATE_FEATURE_NAMES)}
IDX_Z = tuple(EMIT_STATE_INDEX[n] for n in ("z_snow", "z_pv", "z_npv", "z_soil"))

# ---------------------------------------------------------------------------
# Hard-coded EMIT pixel (split_files/emit_data_snow_70to100.nc, row 0)
# ---------------------------------------------------------------------------
EMBEDDED_PIXEL_INDEX = 0
EMBEDDED_SAMPLE_ID = 376
EMBEDDED_SOURCE = "split_files/emit_data_snow_70to100.nc"

EMBEDDED_STATE = np.array([
    0.5737820863723755, 0.34885212779045105, 1348.8544921875, 8.800702095031738, 693.9166259765625,
    86516.140625, 2.1627614498138428, 0.21034644544124603, -3.4946959018707275, -1.8683929443359375,
    -2.99912428855896, -1.0070809125900269, -2.0869531631469727, 0.1337398886680603, 0.49470919370651245,
], dtype=np.float64)

EMBEDDED_REFLECTANCE = np.array([
    0.38285755662943405, 0.37156529693557355, 0.3612893690186621, 0.4006081036337249, 0.38542470765767783, 0.39850801005488706, 0.3965854466583473, 0.38839635920628557,
    0.39743614297870244, 0.40088778782056317, 0.40141237483703046, 0.4017335813460515, 0.39946403831254806, 0.3989245599518698, 0.3955675022889638, 0.3991774427472607,
    0.3951790870371402, 0.3951986054088717, 0.3914675129027711, 0.3900352471115928, 0.38844648380017055, 0.38528804630116414, 0.3817647239136271, 0.3845169336127102,
    0.3743176761435289, 0.37102559247141403, 0.36177376857831306, 0.36861735430291165, 0.36520405916829213, 0.3683898532112065, 0.3706911079170928, 0.3715141836333271,
    0.3739645781580943, 0.36808792089956505, 0.3746962413233157, 0.3847470508454722, 0.38386925443605097, 0.3838417214363526, 0.396057448442066, 0.39549010508751964,
    0.39699435225049, 0.36244307186341795, 0.36887670862269434, 0.39632731513039743, 0.4034108593137458, 0.3930185350624985, 0.3821892346823034, 0.3919099137788247,
    0.40565385444248103, 0.408939816955203, 0.3998961656428873, 0.25432111256111317, 0.32035698613273483, 0.39412702039109, 0.3896798820501154, 0.3845209526963313,
    0.38276356066317024, 0.38541378160460427, 0.37006869672418397, 0.35976529041197236, 0.370676155392981, 0.3788286458985976, 0.3848280461942876, 0.37679557043158335,
    0.37359636413546243, 0.3664172962121554, 0.35909741730846395, 0.34958216455635766, 0.34151208120133236, 0.3205649934901426, 0.30583126066778915, 0.3027046968983015,
    0.31052956683158595, 0.3088000094156614, 0.22397097049584605, 0.21933357158262323, 0.224195167320974, 0.23265372556576905, 0.2447511745781059, 0.25914071398002075,
    0.2500205420273949, 0.24939229230706159, 0.23976498150911318, 0.2303255911391744, 0.2207469424479843, 0.21737757940447927, 0.21269963854832757, 0.21404618662998037,
    0.21684509438363922, 0.22209881407612447, 0.229576740220757, 0.23314954061479284, 0.24128303787859037, 0.2547272723429281, 0.2594003401354585, 0.2612683670894556,
    0.2546352456659441, 0.24744058296054788, 0.2207012514267459, 0.1581930209117632, 0.13594111486159047, 0.14655014518488682, 0.1644434907089661, 0.14341510073679734,
    0.16106869827313436, 0.1713899450573095, 0.16437406685852085, 0.1496938358407681, 0.13705078661801576, 0.12889690621905778, 0.1137017948215209, 0.10681940619874357,
    0.10395382573522122, 0.10053478701935098, 0.10144852511850394, 0.09953116518676637, 0.09795982036885735, 0.09553040728776337, 0.08674134861738629, 0.07920813687804466,
    0.09234334393269227, 0.1010318062575757, 0.11459829783491328, 0.1308358906022887, 0.13865817780121442, 0.126226276432613, 0.09048024495320098, 0.08585722536190946,
    0.07740245926392991, 0.06974669202784299, 0.03406195953498522, 0.013216932365725102, 0.004481705473902615, 0.006662813228804335, 0.007986274825772693, 0.009265084811794484,
    0.010302658299203274, 0.00852960532422578, 0.01370668031853487, 0.017159819536006055, 0.017804236253563317, 0.014344449901503993, 0.012676953815563706, 0.012774509990819522,
    0.014276741420135833, 0.015317268425543111, 0.013166286234939574, 0.015159064089753909, 0.015772939530617, 0.017924588884904655, 0.02114974092890561, 0.021155310679658753,
    0.03716625470076101, 0.037138160133060934, 0.022237161621014436, 0.021999827755433063, 0.022526221802368487, 0.02265281829083728, 0.022263099085675263, 0.022204226758623956,
    0.02032927082265645, 0.021331249132267725, 0.022749281418844646, 0.02284141551012531, 0.02153091211766182, 0.0221493533921327, 0.024012801679461203, 0.0241615454036744,
    0.02400499787021539, 0.024221058268298335, 0.023946103509893887, 0.0251088318685376, 0.024496293467668673, 0.02477263614801578, 0.0257190785158383, 0.026175282957105715,
    0.02616724315077758, 0.026340274752008196, 0.02591944270252623, 0.02635670588754755, 0.026143592453411052, 0.0258260647406421, 0.02684230817434731, 0.026718509076157353,
    0.026869680498214327, 0.02734645833498794, 0.02621765398134089, 0.0266251564677492, 0.02503711188102642, 0.024284488389771364, 0.020557632512757284, 0.01637270957909434,
    0.013642928916261031, 0.009368743191930248, 0.007186191760499819, 0.0014982431301981034, 0.0006564327050819458, 0.0009056805862722851, 0.0007231904766189641, 0.0010354603815348874,
    0.0005459977550726241, 0.0007398229125762609, 0.0014936887728949858, 0.0010440782838373665, 0.0007071348842051334, 0.0008680483204854815, 0.001141701708793135, 0.0024832301737954107,
    0.0036106855656130687, 0.006252096436520347, 0.007499365071847655, 0.006423579079721078, 0.006294079740560662, 0.008140549619584232, 0.015743236253175166, 0.018660721054378496,
    0.015137273856472988, 0.010730012870074385, 0.004809894175591485, 0.0046582173015168055, 0.008328457318005635, 0.013268851850720121, 0.015494014148479198, 0.01518776029134227,
    0.02135302349436779, 0.01965329005014947, 0.01930545698840731, 0.011261917888252588, 0.014274897309609269, 0.015397025301971144, 0.01853109291062895, 0.018730389970735974,
    0.017288481067314766, 0.01698904290540479, 0.016652878941107608, 0.016142906896571427, 0.017191218049710882, 0.01589706435952716, 0.016321012003248998, 0.017334518247548756,
    0.017783233557483603, 0.01848981787323108, 0.018117376123194265, 0.019451670790938694, 0.019068086344925073, 0.019402135617731808, 0.02012562530263252, 0.02038317145752453,
    0.020488643269783245, 0.020318747985967466, 0.01889101738146198, 0.01854560067763592, 0.018628579556564977, 0.017921618704527607, 0.01704246389236706, 0.016903834485912504,
    0.01728449502094541, 0.015475192858443855, 0.014506039576653845, 0.014914536363223842, 0.013193342109544573, 0.012856862309267358, 0.015542955301724354, 0.012656423816176836,
    0.010473504050812783, 0.011653263934890043, 0.013493534333844693, 0.011107459050349489, 0.010710220900984349, 0.011492090248641349, 0.006480547192857836, 0.006873635029759475,
    0.01192563663495161, 0.01034289794672203, 0.01264829582580459, 0.013366173229396493, 0.009835881981519985, 0.01005164318564185, 0.010157832408973185, 0.007848611249599529,
    0.005658954638415263, 0.008037635951200666, 0.004803295888796691, 0.003097934695632855, 0.0024140431956508283,
], dtype=np.float64)

EMBEDDED_WL = np.array([
    381.0055847167969, 388.4092102050781, 395.8158264160156, 403.2254028320313, 410.6380004882813, 418.0535888671875, 425.4721374511719, 432.8927001953125,
    440.3172607421875, 447.7427978515625, 455.17034912109375, 462.598876953125, 470.0303955078125, 477.4629211425781, 484.8974304199219, 492.3329162597656,
    499.77142333984375, 507.20989990234375, 514.650390625, 522.0908813476562, 529.5333251953125, 536.976806640625, 544.4212646484375, 551.86669921875,
    559.314208984375, 566.7615966796875, 574.2090454101562, 581.6585083007812, 589.1079711914062, 596.558349609375, 604.0098266601562, 611.4622192382812,
    618.9146118164062, 626.3680419921875, 633.8214721679688, 641.27587890625, 648.7302856445312, 656.1857299804688, 663.64111328125, 671.0975341796875,
    678.5538940429688, 686.0103149414062, 693.4677124023438, 700.9251098632812, 708.383544921875, 715.8409423828125, 723.29931640625, 730.7587280273438,
    738.2171020507812, 745.676513671875, 753.1359252929688, 760.5963134765625, 768.0557250976562, 775.51611328125, 782.9775390625, 790.4379272460938,
    797.8993530273438, 805.3617553710938, 812.8231811523438, 820.2846069335938, 827.7459716796875, 835.2073974609375, 842.6698608398438, 850.1312866210938,
    857.5936889648438, 865.0551147460938, 872.517578125, 879.9800415039062, 887.4414672851562, 894.9039306640625, 902.3663940429688, 909.828857421875,
    917.2913208007812, 924.7537841796876, 932.2162475585938, 939.6787719726562, 947.1402587890624, 954.6027221679688, 962.0642700195312, 969.5267944335938,
    976.98828125, 984.4498291015624, 991.911376953125, 999.3728637695312, 1006.8344116210938, 1014.2949829101562, 1021.756591796875, 1029.2171630859375,
    1036.677734375, 1044.1383056640625, 1051.598876953125, 1059.0595703125, 1066.5201416015625, 1073.979736328125, 1081.4404296875, 1088.9000244140625,
    1096.3597412109375, 1103.818359375, 1111.278076171875, 1118.73681640625, 1126.1964111328125, 1133.6551513671875, 1141.1129150390625, 1148.5716552734375,
    1156.0303955078125, 1163.4881591796875, 1170.9459228515625, 1178.4036865234375, 1185.861572265625, 1193.318359375, 1200.776123046875, 1208.2330322265625,
    1215.6898193359375, 1223.146728515625, 1230.6036376953125, 1238.0595703125, 1245.515380859375, 1252.972412109375, 1260.4283447265625, 1267.88330078125,
    1275.3392333984375, 1282.794189453125, 1290.250244140625, 1297.7052001953125, 1305.1602783203125, 1312.6143798828125, 1320.0684814453125, 1327.5224609375,
    1334.9755859375, 1342.4287109375, 1349.8818359375, 1357.3350830078125, 1364.7872314453125, 1372.2384033203125, 1379.690673828125, 1387.141845703125,
    1394.5931396484375, 1402.0433349609375, 1409.49365234375, 1416.9439697265625, 1424.393310546875, 1431.8426513671875, 1439.2919921875, 1446.7403564453125,
    1454.1888427734375, 1461.63720703125, 1469.084716796875, 1476.5321044921875, 1483.9796142578125, 1491.4261474609375, 1498.8726806640625, 1506.3192138671875,
    1513.764892578125, 1521.21044921875, 1528.655029296875, 1536.1007080078125, 1543.54541015625, 1550.9891357421875, 1558.432861328125, 1565.8765869140625,
    1573.3193359375, 1580.7620849609375, 1588.2049560546875, 1595.646728515625, 1603.088623046875, 1610.529541015625, 1617.970458984375, 1625.410400390625,
    1632.851318359375, 1640.290283203125, 1647.7303466796875, 1655.16943359375, 1662.607421875, 1670.0455322265625, 1677.483642578125, 1684.9208984375,
    1692.3580322265625, 1699.795166015625, 1707.2314453125, 1714.666748046875, 1722.10302734375, 1729.538330078125, 1736.97265625, 1744.4071044921875,
    1751.8414306640625, 1759.27490234375, 1766.7083740234375, 1774.141845703125, 1781.5743408203125, 1789.0069580078125, 1796.4384765625, 1803.8701171875,
    1811.30078125, 1818.7314453125, 1826.1611328125, 1833.5909423828125, 1841.0206298828125, 1848.449462890625, 1855.8773193359373, 1863.30517578125,
    1870.7330322265625, 1878.1600341796875, 1885.5869140625, 1893.012939453125, 1900.43896484375, 1907.864013671875, 1915.2891845703125, 1922.7132568359373,
    1930.137451171875, 1937.5606689453125, 1944.98388671875, 1952.4071044921875, 1959.8294677734373, 1967.2518310546875, 1974.6732177734373, 1982.0946044921875,
    1989.5150146484373, 1996.935546875, 2004.35498046875, 2011.7745361328125, 2019.193115234375, 2026.61181640625, 2034.0303955078125, 2041.4471435546875,
    2048.864990234375, 2056.28076171875, 2063.696533203125, 2071.1123046875, 2078.52734375, 2085.942138671875, 2093.356201171875, 2100.76904296875,
    2108.18212890625, 2115.59423828125, 2123.00634765625, 2130.41748046875, 2137.828857421875, 2145.239013671875, 2152.648193359375, 2160.0576171875,
    2167.467041015625, 2174.87548828125, 2182.282958984375, 2189.6904296875, 2197.096923828125, 2204.50341796875, 2211.9091796875, 2219.314697265625,
    2226.719482421875, 2234.123291015625, 2241.52685546875, 2248.9296875, 2256.332763671875, 2263.734619140625, 2271.136474609375, 2278.53759765625,
    2285.938720703125, 2293.338623046875, 2300.73779296875, 2308.135986328125, 2315.5341796875, 2322.9326171875, 2330.329833984375, 2337.726318359375,
    2345.12158203125, 2352.51708984375, 2359.91259765625, 2367.30712890625, 2374.70068359375, 2382.093505859375, 2389.486083984375, 2396.8779296875,
    2404.26953125, 2411.660400390625, 2419.05126953125, 2426.440185546875, 2433.830322265625, 2441.21826171875, 2448.6064453125, 2455.994384765625,
    2463.381591796875, 2470.767822265625, 2478.153076171875, 2485.53857421875, 2492.923828125,
], dtype=np.float64)

EMIT_COLOR = "#1f4e79"
SIM_COLOR = "#b85c38"
BEST_COLOR = "#2d6a4f"


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-30, None)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        x = obj.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def state_to_forward_args(state: np.ndarray) -> dict[str, float]:
    """Map 15-D EMIT state to simulate_pixel kwargs (z_* as logits)."""
    state = np.asarray(state, dtype=np.float64).ravel()
    if state.shape[0] != len(EMIT_STATE_FEATURE_NAMES):
        raise ValueError(
            f"Expected state length {len(EMIT_STATE_FEATURE_NAMES)}, got {state.shape[0]}"
        )
    idx = EMIT_STATE_INDEX
    return {
        "grain": float(state[idx["grain_radius"]]),
        "lwc": float(state[idx["liquid_water"]]),
        "dust": float(state[idx["dust"]]),
        "algae": float(state[idx["algae"]]),
        "fsnow": float(state[idx["z_snow"]]),
        "fPV": float(state[idx["z_pv"]]),
        "fNPV": float(state[idx["z_npv"]]),
        "fsoil": float(state[idx["z_soil"]]),
        "cwv": float(state[idx["H20STR"]]),
        "aot": float(state[idx["AOT660"]]),
    }


def state_report(state: np.ndarray) -> dict[str, Any]:
    state = np.asarray(state, dtype=np.float64).ravel()
    forward = state_to_forward_args(state)
    frac = softmax_rows(state[list(IDX_Z)])[0]
    return {
        "raw_state": {name: float(state[i]) for i, name in enumerate(EMIT_STATE_FEATURE_NAMES)},
        "forward_args_logits": {
            k: forward[k]
            for k in ("fsnow", "fPV", "fNPV", "fsoil")
        },
        "forward_args_physical": {
            k: forward[k]
            for k in ("grain", "lwc", "dust", "algae", "cwv", "aot")
        },
        "softmax_fractions": {
            "fsnow": float(frac[0]),
            "fPV": float(frac[1]),
            "fNPV": float(frac[2]),
            "fsoil": float(frac[3]),
        },
    }


def get_embedded_pixel() -> tuple[int, np.ndarray, np.ndarray, int]:
    """Return hard-coded pixel index, state, reflectance, and sample id."""
    state = np.asarray(EMBEDDED_STATE, dtype=np.float64).copy()
    reflectance = np.asarray(EMBEDDED_REFLECTANCE, dtype=np.float64).copy()
    if state.shape[0] != len(EMIT_STATE_FEATURE_NAMES):
        raise ValueError(
            f"EMBEDDED_STATE length {state.shape[0]} != "
            f"{len(EMIT_STATE_FEATURE_NAMES)} feature names"
        )
    if reflectance.shape[0] != EMBEDDED_WL.shape[0]:
        raise ValueError(
            f"EMBEDDED_REFLECTANCE bands {reflectance.shape[0]} != "
            f"EMBEDDED_WL {EMBEDDED_WL.shape[0]}"
        )
    return EMBEDDED_PIXEL_INDEX, state, reflectance, EMBEDDED_SAMPLE_ID


@dataclass
class ForwardModel:
    wl_mod: np.ndarray
    wl_emit: np.ndarray
    ds_dis_wl: np.ndarray
    h_matrix: np.ndarray
    endmembers: np.ndarray
    lut_coszen: float
    solar_irr: np.ndarray
    emit_noise: pd.DataFrame
    v_interp_latm: Any
    v_interp_sphalb: Any
    v_interp_lraw: dict[str, Any]
    v_interp_r_dd: Any
    v_interp_r_hd: Any

    def simulate_pixel(
        self,
        *,
        cos_i: float,
        ele_km: float,
        coszen: float,
        vza: float,
        grain: float,
        lwc: float,
        dust: float,
        algae: float,
        fsnow: float,
        f_pv: float,
        f_npv: float,
        f_soil: float,
        cwv: float,
        aot: float,
        add_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        lookup_pt = np.array(
            [
                np.degrees(np.arccos(cos_i)),
                vza,
                RAA_DISORT,
                grain,
                algae,
                dust,
                lwc,
            ],
            dtype=np.float64,
        )
        rho_dd_22 = self.v_interp_r_dd(lookup_pt)
        rho_hd_22 = self.v_interp_r_hd(lookup_pt)
        rho_dd = np.interp(self.wl_mod, self.ds_dis_wl, rho_dd_22)
        rho_hd = np.interp(self.wl_mod, self.ds_dis_wl, rho_hd_22)

        f = softmax(np.array([fsnow, f_pv, f_npv, f_soil], dtype=np.float64))
        rho_dd = rho_dd * f[0] + np.dot(self.endmembers, f[1:])
        rho_hd = rho_hd * f[0] + np.dot(self.endmembers, f[1:])

        if not (0.0 < coszen <= 1.0):
            raise ValueError(f"coszen must be in (0, 1], got {coszen}")
        atm_pt = np.array(
            [ele_km, 180.0 - vza, RAA_TRUE, aot, cwv], dtype=np.float64
        )
        l_atm = self.v_interp_latm(atm_pt)
        s_alb = self.v_interp_sphalb(atm_pt)
        l_raw = [
            self.v_interp_lraw[k](atm_pt)
            for k in ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
        ]
        solar_irr_emit = np.dot(self.h_matrix, self.solar_irr)

        # ISOFIT 6c: form L_tot before eq. 11 on diffuse terms; residual is
        # (L_tot * s * rho^2) / (1 - s * rho), not L_raw * rho / (1 - s * rho).
        eq_11_term = 1.0 - (s_alb * rho_hd)
        l_dir_dir = (l_raw[0] / coszen) * cos_i
        l_dif_dir = l_raw[1] * (cos_i / coszen)
        l_dir_dif = l_raw[2]  # flat background: cos_i_bg = coszen
        l_dif_dif = l_raw[3]
        l_tot = l_dir_dir + l_dif_dir + l_dir_dif + l_dif_dif
        l_dif_dir = l_dif_dir / eq_11_term
        l_dif_dif = l_dif_dif / eq_11_term
        atm_surface_scattering = s_alb * rho_hd

        toa_rdn = (
            l_atm
            + l_dir_dir * rho_dd
            + l_dif_dir * rho_hd
            + l_dir_dif * rho_hd
            + l_dif_dif * rho_hd
            + (l_tot * atm_surface_scattering * rho_hd) / eq_11_term
        )

        rdn_emit = np.dot(self.h_matrix, toa_rdn)
        if add_noise:
            rdn_out = apply_noise(rdn_emit, self.wl_emit, self.emit_noise)
        else:
            rdn_out = rdn_emit
        toa_ref = rdn_out * np.pi / (solar_irr_emit * coszen)
        return rdn_out, toa_ref


def apply_noise(
    rdn: np.ndarray, wl: np.ndarray, noise_df: pd.DataFrame
) -> np.ndarray:
    a = np.interp(wl, noise_df["wvl"], noise_df["a"])
    b = np.interp(wl, noise_df["wvl"], noise_df["b"])
    c = np.interp(wl, noise_df["wvl"], noise_df["c"])
    nedl = a * np.sqrt(np.maximum(b + rdn, 1e-5)) + c
    sy = np.diagflat(np.power(nedl, 2))
    return rdn + np.random.multivariate_normal(np.zeros(rdn.shape), sy)


def load_forward_model(*, wl_emit: np.ndarray) -> ForwardModel:
    ds_mod = xr.open_zarr(MODTRAN_PATH)
    wl_mod = np.asarray(ds_mod.wl.values, dtype=np.float64)
    target_dims = (
        "surface_elevation_km",
        "observer_zenith",
        "relative_azimuth",
        "AOT550",
        "H2OSTR",
        "wl",
    )
    modtran_grid = [
        ds_mod[k].values
        for k in [
            "surface_elevation_km",
            "observer_zenith",
            "relative_azimuth",
            "AOT550",
            "H2OSTR",
        ]
    ]
    v_interp_latm = VectorInterpolator(
        modtran_grid, ds_mod.rhoatm.transpose(*target_dims).values, version="mlg"
    )
    v_interp_sphalb = VectorInterpolator(
        modtran_grid, ds_mod.sphalb.transpose(*target_dims).values, version="mlg"
    )
    v_interp_lraw = {
        k: VectorInterpolator(
            modtran_grid, ds_mod[k].transpose(*target_dims).values, version="mlg"
        )
        for k in ["dir-dir", "dif-dir", "dir-dif", "dif-dif"]
    }

    ds_dis = xr.load_dataset(DISORT_PATH)
    disort_grid = [
        ds_dis[k].values
        for k in ["sza", "vza", "raa", "grain_radius", "algae_conc", "dust_conc", "lwc"]
    ]
    v_interp_r_dd = VectorInterpolator(disort_grid, ds_dis.r_dd.values, version="mlg")
    v_interp_r_hd = VectorInterpolator(disort_grid, ds_dis.r_hd.values, version="mlg")

    emit_specs = pd.read_csv(
        EMIT_WAVE_PATH, sep=r"\s+", names=["idx", "wl", "fwhm"]
    )
    emit_noise = pd.read_csv(
        NOISE_PATH, sep=r"\s+", names=["wvl", "a", "b", "c", "rmse"], comment="#"
    )
    h_matrix = calculate_resample_matrix(
        wl_mod, emit_specs.wl.values, emit_specs.fwhm.values
    )
    endmembers = np.array(pd.read_csv(ENDMEMBER_PATH))[:, 1:]

    if wl_emit.shape[0] != emit_specs.wl.shape[0]:
        raise ValueError(
            f"wl_emit length {wl_emit.shape[0]} != emit-wave bands {emit_specs.wl.shape[0]}"
        )

    return ForwardModel(
        wl_mod=wl_mod,
        wl_emit=wl_emit,
        ds_dis_wl=np.asarray(ds_dis.wavelength.values, dtype=np.float64),
        h_matrix=h_matrix,
        endmembers=endmembers,
        lut_coszen=float(ds_mod.coszen),
        solar_irr=np.asarray(ds_mod.solar_irr.values, dtype=np.float64),
        emit_noise=emit_noise,
        v_interp_latm=v_interp_latm,
        v_interp_sphalb=v_interp_sphalb,
        v_interp_lraw=v_interp_lraw,
        v_interp_r_dd=v_interp_r_dd,
        v_interp_r_hd=v_interp_r_hd,
    )


def spectrum_metrics(
    sim: np.ndarray, emit: np.ndarray, wl: np.ndarray
) -> dict[str, float | int]:
    sim = np.asarray(sim, dtype=np.float64)
    emit = np.asarray(emit, dtype=np.float64)
    delta = sim - emit
    rmse = float(np.sqrt(np.mean(delta**2)))
    bias = float(np.mean(delta))
    rel = float(np.mean(np.abs(delta) / np.maximum(np.abs(emit), 1e-6)))
    nir_i = int(np.argmin(np.abs(wl - NIR_TARGET_NM)))
    return {
        "rmse": rmse,
        "mean_bias": bias,
        "mean_abs_rel_error": rel,
        "mean_reflectance_sim": float(np.mean(sim)),
        "mean_reflectance_emit": float(np.mean(emit)),
        "nir_band_index": nir_i,
        "nir_wavelength_nm": float(wl[nir_i]),
        "nir_delta": float(delta[nir_i]),
    }


def run_geometry_sweep(
    model: ForwardModel,
    forward_args: dict[str, float],
    cos_i_grid: np.ndarray,
    ele_grid: np.ndarray,
    *,
    coszen: float,
    vza: float,
    add_noise: bool,
) -> np.ndarray:
    n_ele = int(ele_grid.size)
    n_cos = int(cos_i_grid.size)
    n_bands = int(model.wl_emit.size)
    refl_grid = np.empty((n_ele, n_cos, n_bands), dtype=np.float64)
    n_total = n_ele * n_cos
    done = 0
    for i, ele_km in enumerate(ele_grid):
        for j, cos_i in enumerate(cos_i_grid):
            _, toa_ref = model.simulate_pixel(
                cos_i=float(cos_i),
                ele_km=float(ele_km),
                coszen=float(coszen),
                vza=float(vza),
                grain=forward_args["grain"],
                lwc=forward_args["lwc"],
                dust=forward_args["dust"],
                algae=forward_args["algae"],
                fsnow=forward_args["fsnow"],
                f_pv=forward_args["fPV"],
                f_npv=forward_args["fNPV"],
                f_soil=forward_args["fsoil"],
                cwv=forward_args["cwv"],
                aot=forward_args["aot"],
                add_noise=add_noise,
            )
            refl_grid[i, j] = toa_ref
            done += 1
            if done == 1 or done == n_total or done % 50 == 0:
                print(f"  simulated {done}/{n_total}", flush=True)
    return refl_grid


def plot_spectrum_overlay(
    wl: np.ndarray,
    emit_refl: np.ndarray,
    sim_default: np.ndarray,
    sim_best: np.ndarray,
    *,
    cos_default: float,
    ele_default: float,
    coszen_default: float,
    vza_default: float,
    cos_best: float,
    ele_best: float,
    coszen_best: float,
    vza_best: float,
    out_path: Path,
) -> None:
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0], "hspace": 0.08},
    )
    ax0.plot(wl, emit_refl, color=EMIT_COLOR, lw=1.8, label="EMIT observed")
    ax0.plot(
        wl,
        sim_default,
        color=SIM_COLOR,
        lw=1.5,
        ls="--",
        label=(
            f"sim default (cos_i={cos_default:.3f}, coszen={coszen_default:.3f}, "
            f"ele={ele_default:.1f} km, VZA={vza_default:.1f}°)"
        ),
    )
    ax0.plot(
        wl,
        sim_best,
        color=BEST_COLOR,
        lw=1.5,
        label=(
            f"sim best RMSE (cos_i={cos_best:.3f}, coszen={coszen_best:.3f}, "
            f"ele={ele_best:.1f} km, VZA={vza_best:.1f}°)"
        ),
    )
    ax0.set_ylabel("TOA reflectance")
    ax0.set_title("EMIT pixel vs forward-model spectra")
    ax0.legend(frameon=False, fontsize=8, loc="upper right")
    ax0.set_ylim(bottom=0)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    delta = sim_best - emit_refl
    ax1.axhline(0.0, color="0.55", lw=0.8)
    ax1.fill_between(wl, 0.0, delta, where=delta >= 0, color=BEST_COLOR, alpha=0.28, linewidth=0)
    ax1.fill_between(wl, 0.0, delta, where=delta < 0, color=EMIT_COLOR, alpha=0.28, linewidth=0)
    ax1.plot(wl, delta, color="#2b2b2b", lw=1.5, label="best sim − EMIT")
    ax1.set_xlabel("wavelength (nm)")
    ax1.set_ylabel("Δ reflectance")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_delta(
    wl: np.ndarray,
    emit_refl: np.ndarray,
    sim_best: np.ndarray,
    *,
    cos_best: float,
    ele_best: float,
    coszen_best: float,
    vza_best: float,
    rmse: float,
    out_path: Path,
) -> None:
    delta = sim_best - emit_refl
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.plot(wl, delta, color=BEST_COLOR, lw=1.6)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("Δ reflectance")
    ax.set_title(
        f"Best-match residual (RMSE={rmse:.5f}, cos_i={cos_best:.3f}, "
        f"coszen={coszen_best:.3f}, ele={ele_best:.1f} km, VZA={vza_best:.1f}°)"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_heatmap(
    rmse_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    title: str,
    best_x_idx: int,
    best_y_idx: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(
        rmse_grid,
        origin="lower",
        aspect="auto",
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        cmap="viridis",
    )
    ax.plot(
        x_grid[best_x_idx],
        y_grid[best_y_idx],
        marker="*",
        color="white",
        ms=12,
        mec="black",
        mew=0.6,
        label="best RMSE",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RMSE")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_idx, state, emit_refl, sample_id = get_embedded_pixel()
    report = state_report(state)
    fsnow = report["softmax_fractions"]["fsnow"]
    print(
        f"Embedded pixel_index={pixel_idx} sample_id={sample_id} "
        f"source={EMBEDDED_SOURCE} fsnow={fsnow:.3f}"
    )

    wl = np.asarray(EMBEDDED_WL, dtype=np.float64)
    if emit_refl.shape[0] != wl.shape[0]:
        raise ValueError(
            f"EMIT bands {emit_refl.shape[0]} != wavelength grid {wl.shape[0]}"
        )

    print("Loading forward model LUTs...")
    model = load_forward_model(wl_emit=wl)

    forward_args = state_to_forward_args(state)
    cos_i_grid = np.linspace(args.cos_i_min, args.cos_i_max, args.cos_i_n)
    ele_grid = np.linspace(args.ele_min, args.ele_max, args.ele_n)
    coszen_fixed = float(model.lut_coszen)
    vza_fixed = VZA_DEFAULT

    n_total = args.ele_n * args.cos_i_n
    print(f"Fixed coszen (LUT) = {coszen_fixed:.4f}")
    print(f"Fixed VZA = {vza_fixed:.1f}°")
    print(
        f"Sweeping {args.ele_n} x {args.cos_i_n} = {n_total} geometry points..."
    )
    refl_grid = run_geometry_sweep(
        model,
        forward_args,
        cos_i_grid,
        ele_grid,
        coszen=coszen_fixed,
        vza=vza_fixed,
        add_noise=args.add_noise,
    )

    rmse_grid = np.empty((ele_grid.size, cos_i_grid.size), dtype=np.float64)
    bias_grid = np.empty_like(rmse_grid)
    for i in range(ele_grid.size):
        for j in range(cos_i_grid.size):
            metrics = spectrum_metrics(refl_grid[i, j], emit_refl, wl)
            rmse_grid[i, j] = metrics["rmse"]
            bias_grid[i, j] = metrics["mean_bias"]

    best_flat = int(np.nanargmin(rmse_grid))
    best_i, best_j = np.unravel_index(best_flat, rmse_grid.shape)
    cos_best = float(cos_i_grid[best_j])
    ele_best = float(ele_grid[best_i])
    sim_best = refl_grid[best_i, best_j]

    default_j = _nearest_index(cos_i_grid, COS_I_DEFAULT)
    default_i = _nearest_index(ele_grid, ELE_DEFAULT)
    cos_default = float(cos_i_grid[default_j])
    ele_default = float(ele_grid[default_i])
    sim_default = refl_grid[default_i, default_j]

    best_metrics = spectrum_metrics(sim_best, emit_refl, wl)
    default_metrics = spectrum_metrics(sim_default, emit_refl, wl)

    pixel_info = {
        "embedded_source": EMBEDDED_SOURCE,
        "pixel_index": pixel_idx,
        "sample_id": sample_id,
        "n_bands": int(wl.size),
        "fixed_geometry": {
            "RAA": RAA_TRUE,
            "VZA": vza_fixed,
            "coszen": coszen_fixed,
        },
        "lut_coszen": coszen_fixed,
        **report,
    }
    sweep_metrics = {
        "cos_i_grid": cos_i_grid.tolist(),
        "ele_km_grid": ele_grid.tolist(),
        "fixed_coszen": coszen_fixed,
        "fixed_vza": vza_fixed,
        "rmse_grid": rmse_grid.tolist(),
        "mean_bias_grid": bias_grid.tolist(),
        "default_geometry": {
            "cos_i": cos_default,
            "coszen": coszen_fixed,
            "ele_km": ele_default,
            "vza": vza_fixed,
            "metrics": default_metrics,
        },
        "best_geometry": {
            "cos_i": cos_best,
            "coszen": coszen_fixed,
            "ele_km": ele_best,
            "vza": vza_fixed,
            "grid_index": [int(best_i), int(best_j)],
            "metrics": best_metrics,
        },
        "add_noise": bool(args.add_noise),
    }

    with open(out_dir / "pixel_info.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(pixel_info), f, indent=2)
    with open(out_dir / "sweep_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(sweep_metrics), f, indent=2)

    plot_spectrum_overlay(
        wl,
        emit_refl,
        sim_default,
        sim_best,
        cos_default=cos_default,
        ele_default=ele_default,
        coszen_default=coszen_fixed,
        vza_default=vza_fixed,
        cos_best=cos_best,
        ele_best=ele_best,
        coszen_best=coszen_fixed,
        vza_best=vza_fixed,
        out_path=out_dir / "spectrum_overlay.png",
    )
    plot_spectrum_delta(
        wl,
        emit_refl,
        sim_best,
        cos_best=cos_best,
        ele_best=ele_best,
        coszen_best=coszen_fixed,
        vza_best=vza_fixed,
        rmse=best_metrics["rmse"],
        out_path=out_dir / "spectrum_delta_best.png",
    )
    plot_rmse_heatmap(
        rmse_grid,
        cos_i_grid,
        ele_grid,
        x_label="cos_i",
        y_label="elevation (km)",
        title=f"RMSE (coszen={coszen_fixed:.3f}, VZA={vza_fixed:.1f}° fixed)",
        best_x_idx=best_j,
        best_y_idx=best_i,
        out_path=out_dir / "rmse_heatmap_cos_ele.png",
    )

    print(f"Default geometry RMSE={default_metrics['rmse']:.5f}")
    print(
        f"Best geometry cos_i={cos_best:.3f} coszen={coszen_fixed:.3f} "
        f"ele={ele_best:.1f} km VZA={vza_fixed:.1f}° RMSE={best_metrics['rmse']:.5f}"
    )
    print(f"Wrote outputs to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--add-noise", action="store_true")
    p.add_argument("--cos-i-min", type=float, default=0.06)
    p.add_argument("--cos-i-max", type=float, default=0.061)
    p.add_argument("--cos-i-n", type=int, default=25)
    p.add_argument("--ele-min", type=float, default=0.5)
    p.add_argument("--ele-max", type=float, default=5.0)
    p.add_argument("--ele-n", type=int, default=20)
    return p


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

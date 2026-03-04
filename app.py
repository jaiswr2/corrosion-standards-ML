# app.py
# Streamlit app with two tabs:
#   Tab 1: Design Standards + ML (mean-only MC with fixed defaults)
#   Tab 2: ML + Monte Carlo (detailed)
#
# Artifacts expected in repo:
#   artifacts_gpr_k/preprocessor_fitted.joblib
#   artifacts_gpr_k/gpr_k_model.joblib
#   artifacts_gpr_k/metadata.json

import json
import re
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FormatStrFormatter


# =========================
# CONFIG / THEME
# =========================
ARTIFACT_DIR = Path(__file__).parent / "artifacts_gpr_k"
PREP_PATH = ARTIFACT_DIR / "preprocessor_fitted.joblib"
MODEL_PATH = ARTIFACT_DIR / "gpr_k_model.joblib"
META_PATH = ARTIFACT_DIR / "metadata.json"

EPS = 1e-12
MC_NS_DEFAULT = 5000

# McMaster theme
MCMAROON = "#7A003C"
MCYELLOW = "#FDB515"
LABELBLUE = "#1f5fbf"
BLACKTXT = "#111111"

FMT = FormatStrFormatter("%g")

# ML feature ranges (for input bounds)
RANGES = {
    "Soil_pH": (3.0, 10.0),
    "Chloride Content (mg/kg)": (0.3, 11400.0),
    "Soil_Resistivity (Ω·cm)": (80.0, 44000.0),
    "Sulphate_Content (mg/kg)": (6.9, 21800.0),
    "Moisture_Content (%)": (1.7, 261.4),
}

STEPS = {
    "Soil_pH": 0.1,
    "Chloride Content (mg/kg)": 50.0,
    "Soil_Resistivity (Ω·cm)": 100.0,
    "Sulphate_Content (mg/kg)": 100.0,
    "Moisture_Content (%)": 10.0,
    "Temperature (°C)": 1.0,
}

SOIL_TYPES = ["GT", "CL", "SM", "ML", "SP", "CH", "GP", "SW", "OL", "SC"]
WATER_TABLE = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]
FOREIGN_INCL = ["None", "Shreded wood", "Cinder", "Flyash"]  # keep spelling consistent with training
FILL_MATERIAL = [0, 1]

AGES_HORIZON = [10, 30, 50, 70, 80]


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    missing = [p for p in [PREP_PATH, MODEL_PATH, META_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts. Ensure the repo contains:\n"
            + "\n".join([f"- {p}" for p in missing])
        )
    prep = joblib.load(PREP_PATH)
    gpr = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return prep, gpr, meta


# =========================
# COMMON HELPERS
# =========================
def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def fmt_ci(lo, hi, nd=3):
    return f"{lo:.{nd}f} – {hi:.{nd}f}"


def count_missing_ml_features(row_dict, expected_cols):
    miss = 0
    for c in expected_cols:
        v = row_dict.get(c, None)
        if v is None:
            miss += 1
        elif isinstance(v, float) and np.isnan(v):
            miss += 1
    return miss


def derive_foreign_flag(foreign_type_value) -> int:
    """No separate toggle. If Foreign_Inclusion_Type != 'None', treat as foreign inclusion present."""
    if foreign_type_value is None:
        return 0
    try:
        s = str(foreign_type_value).strip().lower()
    except Exception:
        return 0
    return 0 if s == "none" else 1


# =========================
# MONTE CARLO HELPERS (ML)
# =========================
def rtruncnorm(mean, sd, low, high, size, seed=42, max_iter=5_000_000):
    rng = np.random.default_rng(seed)
    out = np.empty(size, dtype=float)
    filled = 0
    it = 0
    while filled < size:
        batch = min(size - filled, 100000)
        draw = rng.normal(mean, sd, size=batch)
        draw = draw[(draw >= low) & (draw <= high)]
        k = len(draw)
        if k > 0:
            out[filled : filled + k] = draw[:k]
            filled += k
        it += batch
        if it > max_iter:
            raise RuntimeError("TruncNormal rejection did not converge. Check sd/bounds.")
    return out


def mc_TL_from_k(
    mu_k,
    sd_k,
    ages,
    T_used,
    T0,
    mu_n,
    mu_beta,
    n_bounds=(0.4, 0.7),
    beta_bounds=(0.02, 0.04),
    Ns=5000,
    seed=42,
    shared_n_beta=False,
):
    """
    k ~ Normal(mu_k, sd_k). Clip k at EPS to keep TL physical.
    n, beta ~ TruncNormal using bounds as ~95% interval.
    TL = k * t^n * exp(beta*(T - T0))

    Returns DataFrame with Mean_TL, TL_sd and CI bounds.
    """
    rng = np.random.default_rng(seed)
    mu_k = float(mu_k)
    sd_k = float(max(sd_k, EPS))

    nL, nU = n_bounds
    bL, bU = beta_bounds
    sigma_n = (nU - nL) / (2.0 * 1.96)
    sigma_beta = (bU - bL) / (2.0 * 1.96)

    ages = np.asarray(ages, float)
    out_rows = []

    for t_age in ages:
        z = rng.standard_normal(Ns)
        k_s = np.maximum(mu_k + sd_k * z, EPS)

        if shared_n_beta:
            n_draw = rtruncnorm(mu_n, sigma_n, nL, nU, Ns, seed=seed + 1)
            b_draw = rtruncnorm(mu_beta, sigma_beta, bL, bU, Ns, seed=seed + 2)
        else:
            # still independent draws (same sizes); kept for parity with your prior code
            n_draw = rtruncnorm(mu_n, sigma_n, nL, nU, Ns, seed=seed + 1)
            b_draw = rtruncnorm(mu_beta, sigma_beta, bL, bU, Ns, seed=seed + 2)

        time_fac = np.power(max(float(t_age), EPS), n_draw)
        temp_fac = np.exp(b_draw * (float(T_used) - float(T0)))
        TL_s = k_s * time_fac * temp_fac

        TL_mean = float(np.mean(TL_s))
        TL_sd = float(np.std(TL_s, ddof=1))

        lo68 = max(TL_mean - TL_sd, 0.0)
        hi68 = TL_mean + TL_sd
        lo95 = max(TL_mean - 2.0 * TL_sd, 0.0)
        hi95 = TL_mean + 2.0 * TL_sd

        out_rows.append(
            {
                "Age": int(t_age),
                "Mean_TL (mm)": TL_mean,
                "TL_sd (mm)": TL_sd,
                "TL_lo68 (mm)": lo68,
                "TL_hi68 (mm)": hi68,
                "TL_lo95 (mm)": lo95,
                "TL_hi95 (mm)": hi95,
            }
        )

    return pd.DataFrame(out_rows)


# =========================
# DESIGN STANDARDS (CODE-BASED)
# NOTE: reusing your logic; we add CSA S6 (2025)
# =========================
REL_TOL = 0.20

STANDARD_KEYS = [
    "CSA_S6",
    "Eurocode",
    "AS2159",
    "NZS",
    "WSDOT",
    "FDOT",
    "Japan",
    "China",
    "DIN",
    "Caltrans",
]

DISPLAY_NAMES = {
    "CSA_S6": "CSA S6:2025 (Canada)",
    "Eurocode": "EN 1993-5:2007 (Eurocode)",
    "AS2159": "AS 2159:2009 (Australia)",
    "NZS": "NZS 3404-1:2009 (New Zealand)",
    "WSDOT": "WSDOT BDM:2020 (USA)",
    "FDOT": "FDOT SDG:2023 (USA)",
    "Japan": "OCDI:2020 (Japan)",
    "China": "JTG 3363:2019 (China)",
    "DIN": "DIN 50929-3:2018 (Germany)",
    "Caltrans": "Caltrans BDM:2025 (USA)",
}

SHORT_NAMES = {
    "CSA_S6": "CSA",
    "Eurocode": "EN",
    "AS2159": "AS",
    "NZS": "NZS",
    "WSDOT": "WSDOT",
    "FDOT": "FDOT",
    "Japan": "OCDI",
    "China": "JTG",
    "DIN": "DIN",
    "Caltrans": "Caltrans",
}

STANDARD_COLORS = {
    "CSA_S6": "#7A003C",
    "Eurocode": "#000000",
    "AS2159": "#0072B2",
    "NZS": "#009E73",
    "WSDOT": "#D55E00",
    "FDOT": "#E69F00",
    "Japan": "#CC79A7",
    "China": "#8B4513",
    "DIN": "#800080",
    "Caltrans": "#56B4E9",
}

# conceptual input names used by standard functions
COL_AGE = "Age (yr)"
COL_PH = "Soil_pH"
COL_CL = "Chloride Content (mg/kg)"
COL_SO4 = "Sulphate_Content (mg/kg)"
COL_RHO = "Soil_Resistivity (Ω·cm)"
COL_SOIL = "Soil Type"
COL_LOC = "Location wrt Water Table"
COL_FILL = "Is_Fill_Material"
COL_FOREIGN = "Has_Foreign_Inclusions"
COL_FTYPE = "Foreign_Inclusion_Type"


def to_num(x):
    if isinstance(x, str):
        x = x.replace(" ", "")
    return pd.to_numeric(x, errors="coerce")


def truthy01(v):
    if pd.isna(v):
        return 0
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "yes", "true", "y"}:
            return 1
        try:
            return 1 if float(t) == 1 else 0
        except Exception:
            return 0
    try:
        return 1 if float(v) == 1 else 0
    except Exception:
        return 0


# ---------- CSA S6 (2025) ----------
CSA_NONAGG_RATE = 0.9 / 75.0  # 0.012 mm/yr (linearized)

def csa_s6_classify_and_rate(row):
    """
    CSA S6 (2025):
    Non-aggressive only if ALL satisfied:
      pH 5-10
      Chloride <= 100 ppm
      Resistivity >= 3000 Ω·cm
      Sulphate <= 200 ppm
      Organic <= 1% (NOT provided -> assumed PASS)
    Note: if resistivity >= 5000, chloride + sulphate limits may be waived.
    Aggressive otherwise -> site-specific assessment (rate = NaN).
    """
    ph = to_num(row.get(COL_PH))
    cl = to_num(row.get(COL_CL))
    so4 = to_num(row.get(COL_SO4))
    rho = to_num(row.get(COL_RHO))

    organic_pass = True  # per your instruction (assumed)
    if pd.isna(ph) or pd.isna(rho):
        # cannot verify -> treat as aggressive
        return "Aggressive (C5–CX)", np.nan, "Insufficient inputs to verify non-aggressive criteria."

    # resistivity waiver for chloride & sulphate
    waiver = (rho >= 5000)

    ok_ph = (5.0 <= ph <= 10.0)
    ok_rho = (rho >= 3000.0)
    ok_cl = True if waiver else (pd.notna(cl) and cl <= 100.0)
    ok_so4 = True if waiver else (pd.notna(so4) and so4 <= 200.0)
    ok_org = organic_pass

    if ok_ph and ok_rho and ok_cl and ok_so4 and ok_org:
        note = "Non-aggressive (C4)."
        if waiver:
            note += " Resistivity ≥5000 Ω·cm → chloride/sulphate limits waived."
        return "Non-aggressive (C4)", CSA_NONAGG_RATE, note

    note = "Aggressive (C5–CX): CSA directs site-specific corrosion assessment by a corrosion specialist."
    return "Aggressive (C5–CX)", np.nan, note


# ---------- Eurocode ----------
EC_RATES = {
    "Non-compacted, aggressive fills": 0.057,
    "Non-compacted, non-aggressive fills": 0.022,
    "Aggressive natural soils": 0.032,
    "Polluted natural soils / industrial": 0.030,
    "Undisturbed natural soils": 0.012,
}

def eurocode_class(row):
    fill = truthy01(row.get(COL_FILL))
    foreign = truthy01(row.get(COL_FOREIGN))
    ph = to_num(row.get(COL_PH))
    cl = to_num(row.get(COL_CL))
    so4 = to_num(row.get(COL_SO4))

    if fill == 1:
        return "Non-compacted, aggressive fills" if foreign == 1 else "Non-compacted, non-aggressive fills"
    if pd.notna(ph) and ph < 6:
        return "Aggressive natural soils"
    if (pd.notna(cl) and cl > 300) or (pd.notna(so4) and so4 > 1000):
        return "Polluted natural soils / industrial"
    return "Undisturbed natural soils"

def predict_eurocode(row):
    cls = eurocode_class(row)
    return EC_RATES.get(cls, np.nan)


# ---------- AS 2159 ----------
WT_FLUCT = "fluctuation zone"
WT_IMMERS = "permanent immersion"

AS_RATES = {
    "Non-aggressive": 0.005,
    "Mild": 0.015,
    "Moderate": 0.030,
    "Severe": 0.070,
    "Very Severe": 0.100,
}

def first_token(soil):
    if not isinstance(soil, str) or not soil.strip():
        return None
    return re.split(r"[+/, \t-]+", soil)[0].upper()

def soil_is_coarse(first):
    coarse = {"GW", "GP", "SW", "SP", "SM", "GM", "S", "G", "SC"}
    return first in coarse

def loc_is_below_WT(txt):
    if not isinstance(txt, str):
        return False
    t = txt.lower().strip()
    return (t == WT_FLUCT) or (t == WT_IMMERS)

def pH_bin(ph):
    if pd.isna(ph): return "pH_unk"
    if ph < 4: return "pH_<4"
    if ph < 5: return "pH_4_5"
    if ph < 8.5: return "pH_5_8p5"
    return "pH_>=8p5"

def Cl_bin(cl):
    if pd.isna(cl): return "Cl_unk"
    if cl < 5000: return "Cl_<5k"
    if cl < 20000: return "Cl_5k_20k"
    return "Cl_>=20k"

def upgrade_one(level):
    order = ["Non-aggressive", "Mild", "Moderate", "Severe", "Very Severe"]
    try:
        i = order.index(level)
        return order[min(i + 1, len(order) - 1)]
    except Exception:
        return level

BASE_COND_A = {
    ("pH_<4", "Cl_<5k"): "Moderate",
    ("pH_<4", "Cl_5k_20k"): "Severe",
    ("pH_<4", "Cl_>=20k"): "Very Severe",
    ("pH_4_5", "Cl_<5k"): "Mild",
    ("pH_4_5", "Cl_5k_20k"): "Moderate",
    ("pH_4_5", "Cl_>=20k"): "Severe",
    ("pH_5_8p5", "Cl_<5k"): "Non-aggressive",
    ("pH_5_8p5", "Cl_5k_20k"): "Mild",
    ("pH_5_8p5", "Cl_>=20k"): "Moderate",
    ("pH_>=8p5", "Cl_<5k"): "Non-aggressive",
    ("pH_>=8p5", "Cl_5k_20k"): "Mild",
    ("pH_>=8p5", "Cl_>=20k"): "Moderate",
}

BASE_COND_B = {
    ("pH_<4", "Cl_<5k"): "Moderate",
    ("pH_<4", "Cl_5k_20k"): "Severe",
    ("pH_<4", "Cl_>=20k"): "Very Severe",
    ("pH_4_5", "Cl_<5k"): "Non-aggressive",
    ("pH_4_5", "Cl_5k_20k"): "Non-aggressive",
    ("pH_4_5", "Cl_>=20k"): "Mild",
    ("pH_5_8p5", "Cl_<5k"): "Non-aggressive",
    ("pH_5_8p5", "Cl_5k_20k"): "Mild",
    ("pH_5_8p5", "Cl_>=20k"): "Moderate",
    ("pH_>=8p5", "Cl_<5k"): "Non-aggressive",
    ("pH_>=8p5", "Cl_5k_20k"): "Mild",
    ("pH_>=8p5", "Cl_>=20k"): "Moderate",
}

def classify_as2159_row(row):
    if truthy01(row.get(COL_FOREIGN)) == 1:
        return "Very Severe"
    fi = str(row.get(COL_FTYPE, "")).lower()
    if any(k in fi for k in ["cinder", "fly", "wood"]):
        return "Very Severe"

    first = first_token(row.get(COL_SOIL))
    belowWT = loc_is_below_WT(str(row.get(COL_LOC, "")))
    cond = "A" if (belowWT or soil_is_coarse(first)) else "B"

    phb = pH_bin(to_num(row.get(COL_PH)))
    clb = Cl_bin(to_num(row.get(COL_CL)))

    exposure = (BASE_COND_A if cond == "A" else BASE_COND_B).get((phb, clb), "Non-aggressive")

    so4 = to_num(row.get(COL_SO4))
    if pd.notna(so4) and so4 > 1000:
        exposure = upgrade_one(exposure)

    return exposure

def predict_as2159(row):
    cls = classify_as2159_row(row)
    return AS_RATES.get(cls, np.nan)


# ---------- NZS ----------
NZS_RATES = {
    "Buried in fill below WT": 0.015,
    "Buried in controlled fill above WT": 0.015,
    "Buried in uncontrolled fill above WT (pH≥4)": 0.050,
    "Buried in uncontrolled fill above WT (pH<4)": 0.075,
    "Buried in rubble fill (concrete/brick/inorganic)": 0.025,
    "Undisturbed natural soil": 0.015,
}

def is_below_WT(txt):
    if not isinstance(txt, str): return False
    t = txt.lower()
    return ("below" in t) or ("immersion" in t) or ("fluctuation" in t)

def is_rubble(txt):
    if not isinstance(txt, str): return False
    t = txt.lower()
    return ("concrete" in t) or ("brick" in t) or ("rubble" in t)

def classify_nzs_row(row):
    ph = to_num(row.get(COL_PH))
    fill = truthy01(row.get(COL_FILL))
    foreign = truthy01(row.get(COL_FOREIGN))
    loc = row.get(COL_LOC, "")
    soil = row.get(COL_SOIL, "")

    if fill == 1:
        if is_below_WT(loc):
            cls = "Buried in fill below WT"
            return cls, NZS_RATES[cls]
        if foreign == 1:
            if pd.notna(ph) and ph < 4:
                cls = "Buried in uncontrolled fill above WT (pH<4)"
            else:
                cls = "Buried in uncontrolled fill above WT (pH≥4)"
            return cls, NZS_RATES[cls]
        if is_rubble(soil):
            cls = "Buried in rubble fill (concrete/brick/inorganic)"
            return cls, NZS_RATES[cls]
        cls = "Buried in controlled fill above WT"
        return cls, NZS_RATES[cls]

    cls = "Undisturbed natural soil"
    return cls, NZS_RATES[cls]

def predict_nzs(row):
    _, rate = classify_nzs_row(row)
    return rate


# ---------- WSDOT ----------
WSDOT_RATES = {
    "Undisturbed (non-corrosive)": 0.01270,
    "Undisturbed (corrosive)": 0.02540,
    "Fill/disturbed (non-corrosive)": 0.01905,
    "Fill/disturbed (corrosive)": 0.03810,
}

def is_corrosive_ws(ph, cl, so4):
    if pd.notna(cl) and cl >= 500: return True
    if pd.notna(so4) and so4 >= 1500: return True
    if pd.notna(ph) and ph <= 5.5: return True
    return False

def classify_ws(row):
    ph = to_num(row.get(COL_PH))
    cl = to_num(row.get(COL_CL))
    so4 = to_num(row.get(COL_SO4))
    disturbed = truthy01(row.get(COL_FILL))
    corr = is_corrosive_ws(ph, cl, so4)

    if disturbed == 1:
        name = "Fill/disturbed (corrosive)" if corr else "Fill/disturbed (non-corrosive)"
    else:
        name = "Undisturbed (corrosive)" if corr else "Undisturbed (non-corrosive)"
    return name, WSDOT_RATES[name]

def predict_wsdot(row):
    _, rate = classify_ws(row)
    return rate


# ---------- FDOT ----------
FDOT_RATES = {
    ("Slightly aggressive", "Partially buried"): 0.038,
    ("Slightly aggressive", "Completely buried"): 0.025,
    ("Moderately aggressive", "Partially buried"): 0.051,
    ("Moderately aggressive", "Completely buried"): 0.038,
    ("Extremely aggressive", "Partially buried"): 0.064,
    ("Extremely aggressive", "Completely buried"): 0.051,
}

def burial_from_location(txt):
    if isinstance(txt, str) and ("fluctuation" in txt.lower()):
        return "Partially buried"
    return "Completely buried"

def fdot_aggr(ph, rho, cl):
    extreme = False
    if pd.notna(ph) and ph < 6: extreme = True
    if pd.notna(rho) and rho < 1000: extreme = True
    if pd.notna(cl) and cl >= 2000: extreme = True
    if extreme: return "Extremely aggressive"

    moderate = False
    if pd.notna(ph) and 6 <= ph <= 7: moderate = True
    if pd.notna(rho) and 1000 <= rho <= 5000: moderate = True
    if pd.notna(cl) and 500 <= cl < 2000: moderate = True
    if moderate: return "Moderately aggressive"

    return "Slightly aggressive"

def predict_fdot(row):
    ph = to_num(row.get(COL_PH))
    cl = to_num(row.get(COL_CL))
    rho = to_num(row.get(COL_RHO))
    ag = fdot_aggr(ph, rho, cl)
    buri = burial_from_location(row.get(COL_LOC))
    return FDOT_RATES.get((ag, buri), np.nan)


# ---------- Japan ----------
JAPAN_RATES = {
    "Above residual water level": 0.030,
    "Below residual water level": 0.020,
}

def classify_japan_location(txt):
    if isinstance(txt, str):
        t = txt.lower()
        if ("above" in t) or ("fluctuation" in t):
            return "Above residual water level", JAPAN_RATES["Above residual water level"]
        if ("below" in t) or ("permanent" in t):
            return "Below residual water level", JAPAN_RATES["Below residual water level"]
    return "Below residual water level", JAPAN_RATES["Below residual water level"]

def predict_japan(row):
    _, rate = classify_japan_location(row.get(COL_LOC, ""))
    return rate


# ---------- China ----------
CHINA_RATES = {"Above / Fluctuation": 0.060, "Below": 0.030}

def classify_china_location(txt):
    if not isinstance(txt, str) or txt.strip() == "":
        return "Above / Fluctuation", CHINA_RATES["Above / Fluctuation"]
    t = txt.lower()
    if ("permanent" in t) or ("immersion" in t):
        return "Below", CHINA_RATES["Below"]
    return "Above / Fluctuation", CHINA_RATES["Above / Fluctuation"]

def predict_china(row):
    _, rate = classify_china_location(row.get(COL_LOC, ""))
    return rate


# ---------- Caltrans ----------
CALTRANS_RATES = {
    "Natural Soil": 0.025,
    "Fill/Disturbed": 0.0381,
    "Highly Corrosive Fill": np.nan,  # site-specific
    "Not Corrosive": 0.0,
}

def caltrans_classification(row):
    foreign = int(row.get(COL_FOREIGN, 0)) if pd.notna(row.get(COL_FOREIGN, 0)) else 0
    fill = int(row.get(COL_FILL, 0)) if pd.notna(row.get(COL_FILL, 0)) else 0

    ph = to_num(row.get(COL_PH))
    cl = to_num(row.get(COL_CL))
    so4 = to_num(row.get(COL_SO4))

    if foreign == 1:
        return "Highly Corrosive Fill", CALTRANS_RATES["Highly Corrosive Fill"]

    corrosive = False
    if (pd.notna(ph) and ph <= 5.5) or (pd.notna(cl) and cl >= 500) or (pd.notna(so4) and so4 >= 1500):
        corrosive = True

    if corrosive:
        if fill == 1:
            return "Fill/Disturbed", CALTRANS_RATES["Fill/Disturbed"]
        return "Natural Soil", CALTRANS_RATES["Natural Soil"]

    return "Not Corrosive", CALTRANS_RATES["Not Corrosive"]

def predict_caltrans(row):
    cls, rate = caltrans_classification(row)
    return rate


# ---------- DIN ----------
FOREIGN_KEYWORDS = [
    r"\bfly\s*ash\b", r"\bflyash\b", r"\bash(es)?\b",
    r"\bwood\b", r"\bshred+ed?\s*wood\b", r"\bshredd?ed\b",
    r"\brubble\b", r"\bslag\b", r"\bpeat\b", r"\bfen\b",
    r"\bmud\b", r"\bmarsh\b", r"\brefuse\b", r"\bwaste\b"
]

def has_foreign_inclusion_from_text(soil_type_text):
    if not isinstance(soil_type_text, str) or not soil_type_text.strip():
        return False
    s = soil_type_text.lower()
    for pat in FOREIGN_KEYWORDS:
        if re.search(pat, s):
            return True
    if any(sep in s for sep in "+/,;"):
        parts = re.split(r"[+/;,]", s)
        return any(has_foreign_inclusion_from_text(p) for p in parts)
    return False

def first_uscs_symbol(text):
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.upper().replace("\\", "/")
    for sep in ("+", "/", ",", ";", " "):
        if sep in s:
            s = s.split(sep)[0]
            break
    return s.strip()

def z1_from_uscs_autoflag(soil_type_text, explicit_flag=None):
    base = 0
    sym = first_uscs_symbol(soil_type_text)
    if sym:
        if sym in {"GW", "GP", "SW", "SP"}:
            base = +4
        elif "-" in sym:
            base = +2
        elif sym in {"GM", "GC", "SM", "SC"}:
            base = +2
        elif sym in {"ML", "CL", "MH", "CH", "OL", "OH"}:
            base = -2
        elif sym == "PT":
            base = -2
    contam = bool(explicit_flag) or has_foreign_inclusion_from_text(soil_type_text)
    return base + (-12 if contam else 0)

def z2_from_resistivity_ohm_cm(r):
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return 0
    r = float(r)
    if r > 50000: return +4
    if r > 20000: return +2
    if r > 5000:  return 0
    if r > 2000:  return -2
    if r > 1000:  return -4
    return -6

def z3_from_moisture(m):
    # If not provided -> assume <=20% -> Z3=0
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return 0
    return -1 if float(m) > 20.0 else 0

def z4_from_ph(p):
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return 0
    p = float(p)
    if p > 9.0: return +2
    if p >= 6.0: return 0
    if p >= 4.0: return -1
    return -3

def mmol_per_kg_from_mg_per_kg(mg_per_kg, mm):
    if mg_per_kg is None or (isinstance(mg_per_kg, float) and pd.isna(mg_per_kg)):
        return 0.0
    return float(mg_per_kg) / mm

def z8_from_sulfate_mgkg_acid_extract(so4_mgkg):
    so4 = mmol_per_kg_from_mg_per_kg(so4_mgkg, 96.06)
    if so4 < 2: return 0
    if so4 <= 5: return -1
    if so4 <= 10: return -2
    return -3

def z9_from_neutral_salts(cl_mgkg, so4_mgkg):
    c_cl = mmol_per_kg_from_mg_per_kg(cl_mgkg, 35.45)
    c_so4 = mmol_per_kg_from_mg_per_kg(so4_mgkg, 96.06)
    c_ns = c_cl + 2.0 * c_so4
    if c_ns < 3: return 0
    if c_ns <= 10: return -1
    if c_ns <= 30: return -2
    if c_ns <= 100: return -3
    return -4

def z10_from_water_table(s):
    if not isinstance(s, str): return 0
    s = s.strip().lower()
    if "fluct" in s: return -2
    if "perm" in s or "immer" in s or "below" in s: return -1
    return 0

def z13_from_flags(explicit_flag, soil_text):
    return -6 if (bool(explicit_flag) or has_foreign_inclusion_from_text(soil_text)) else 0

def din_rate_bin_from_total(total):
    if total >= 0: return 0.005, 0.03
    if -4 <= total <= -1: return 0.01, 0.05
    if -10 <= total <= -5: return 0.02, 0.20
    return 0.06, 0.40

def predict_din_uniform(row):
    soil_text = row.get(COL_SOIL)
    explicit_contam = truthy01(row.get(COL_FOREIGN))
    ph = to_num(row.get(COL_PH))
    cl = to_num(row.get(COL_CL))
    rho = to_num(row.get(COL_RHO))
    so4 = to_num(row.get(COL_SO4))
    loc = row.get(COL_LOC)

    moisture = None  # per your earlier comment; if you want, wire Moisture_Content later

    z1 = z1_from_uscs_autoflag(soil_text, explicit_flag=explicit_contam)
    z2 = z2_from_resistivity_ohm_cm(rho)
    z3 = z3_from_moisture(moisture)
    z4 = z4_from_ph(ph)
    z5 = 0
    z6 = 0
    z8 = z8_from_sulfate_mgkg_acid_extract(so4)
    z9 = z9_from_neutral_salts(cl, so4)
    z10 = z10_from_water_table(loc)
    z11 = 0
    z12 = 0
    z13 = z13_from_flags(explicit_contam, soil_text)
    z14 = 0

    b0 = z1 + z2 + z3 + z4 + z5 + z6 + z8 + z9 + z10
    b1 = b0 + z11 + z12 + z13 + z14
    w, _ = din_rate_bin_from_total(b1)
    return w


def compute_all_standards(row):
    rates = {}
    notes = {}

    # CSA
    csa_cls, csa_rate, csa_note = csa_s6_classify_and_rate(row)
    rates["CSA_S6"] = csa_rate
    notes["CSA_S6"] = f"{csa_cls}. {csa_note}"

    # others
    rates["Eurocode"] = predict_eurocode(row)
    rates["AS2159"] = predict_as2159(row)
    rates["NZS"] = predict_nzs(row)
    rates["WSDOT"] = predict_wsdot(row)
    rates["FDOT"] = predict_fdot(row)
    rates["Japan"] = predict_japan(row)
    rates["China"] = predict_china(row)
    rates["DIN"] = predict_din_uniform(row)

    cal_cls, cal_rate = caltrans_classification(row)
    rates["Caltrans"] = cal_rate
    if np.isnan(cal_rate) and cal_cls == "Highly Corrosive Fill":
        notes["Caltrans"] = "Highly Corrosive Fill: Caltrans recommends site-specific assessment."
    else:
        notes["Caltrans"] = cal_cls

    return rates, notes


# =========================
# STREAMLIT UI (TWO TABS)
# =========================
st.set_page_config(page_title="Predicting corrosion-induced thickness loss", layout="wide")

# CSS (kept close to your latest style; consistent labels)
st.markdown(
    f"""
<style>
  .stApp {{ font-size: 60px; }}

  .title {{
    color: {MCMAROON};
    font-weight: 900;
    font-size: 52px;
    margin-bottom: 6px;
  }}

  .subtitle {{
    color: {BLACKTXT};
    font-size: 20px;
    font-weight: 300;
    line-height: 1.25;
    margin-bottom: 6px;
  }}

  .sectiontitle {{
    color: {MCMAROON};
    font-size: 34px;
    font-weight: 900;
    margin-top: 10px;
    margin-bottom: 0px;
  }}

  .sectionnote {{
    color: {BLACKTXT};
    font-size: 22px;
    font-weight: 500;
    margin-top: 4px;
    margin-bottom: 14px;
  }}

  .feat {{
    color: {LABELBLUE};
    font-weight: 700;
    font-size: 18px;
    margin-top: 8px;
    margin-bottom: 4px;
  }}

  div[data-testid="stNumberInput"] label,
  div[data-testid="stSelectbox"] label,
  div[data-testid="stCheckbox"] label {{
    font-size: 16px !important;
    font-weight: 600 !important;
    color: {BLACKTXT} !important;
  }}

  div[data-testid="stCheckbox"] p {{
    font-size: 16px !important;
  }}

  .stNumberInput input,
  .stSelectbox div[data-baseweb="select"] {{
    font-size: 16px !important;
  }}

  /* Yellow buttons (including form submit buttons) */
  div.stButton > button,
  div[data-testid="stFormSubmitButton"] button {{
    background: {MCYELLOW} !important;
    border: 2px solid {MCMAROON} !important;
    color: {MCMAROON} !important;
    font-weight: 900 !important;
    font-size: 20px !important;
    padding: 0.65rem 1.3rem !important;
    border-radius: 12px !important;
  }}
  div.stButton > button:hover,
  div[data-testid="stFormSubmitButton"] button:hover {{
    background: #ffd36a !important;
  }}

  .outline {{
    font-size: 24px;
    line-height: 1.35;
  }}

  .katex-display {{
    margin: 0.4em 0 0.2em 0 !important;
  }}

  /* Tabs: increase font size (robust) */
  div[data-baseweb="tab-list"] button,
  div[data-baseweb="tab-list"] button * {{
    font-size: 54px !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
  }}

  div[data-baseweb="tab-list"] button {{
    padding: 14px 22px !important;
  }}
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    "<div class='title'>Predicting corrosion-induced thickness loss in buried steel pile</div>",
    unsafe_allow_html=True,
)

# NEW subtitle (requested)
st.markdown(
    "<div class='subtitle'>Estimation of uniform corrosion thickness loss of buried steel pile using multiple standards and probablistic ML model</div>",
    unsafe_allow_html=True,
)

prep, gpr, meta = load_artifacts()
expected_cols = meta["expected_raw_columns"]
T0_meta = float(meta["constants"]["T0"])
mu_n_meta = float(meta["constants"]["mu_n"])
mu_beta_meta = float(meta["constants"]["mu_beta"])

# shared defaults in session_state
def _init_state():
    defaults = {
        "Age (yr)": 34,
        "Temperature (°C)": T0_meta,
        "temp_is_na": True,  # default temperature NA => use 10
        # ML features
        "Soil_pH": 7.8,
        "Chloride Content (mg/kg)": 444.0,
        "Soil_Resistivity (Ω·cm)": 900.0,
        "Sulphate_Content (mg/kg)": 328.0,
        "Moisture_Content (%)": 15.0,
        "Soil Type": "CL",
        "Location wrt Water Table": "Above WaterTable",
        "Foreign_Inclusion_Type": "None",
        "Is_Fill_Material": 0,
        # NA flags for ML (default False)
        "na_Soil_pH": False,
        "na_Chloride": False,
        "na_Resistivity": False,
        "na_Sulphate": False,
        "na_Moisture": False,
        "na_SoilType": False,
        "na_WT": False,
        "na_Foreign": False,
        "na_Fill": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def feature_header(text: str):
    st.markdown(f"<div class='feat'>{text}</div>", unsafe_allow_html=True)


def checkbox_unknown(key, default=False, disabled=False):
    return st.checkbox("Unknown (NA)", value=default, key=key, disabled=disabled)


def num_input_no_label(value_key, minv, maxv, default, step, disabled=False):
    return st.number_input(
        label="",
        value=float(default),
        min_value=float(minv),
        max_value=float(maxv),
        step=float(step),
        key=value_key,
        disabled=disabled,
        label_visibility="collapsed",
    )


def select_input_no_label(value_key, options, default_idx=0, disabled=False):
    return st.selectbox(
        label="",
        options=options,
        index=default_idx,
        key=value_key,
        disabled=disabled,
        label_visibility="collapsed",
    )


# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Design Standards + ML", "Detailed ML + Monte Carlo"])


# ============================================================
# TAB 1: STANDARDS + ML (MC mean with fixed defaults)
# ============================================================
with tab1:
    st.markdown("<div class='sectiontitle'>Input Parameters</div>", unsafe_allow_html=True)
    st.markdown("<div class='sectionnote'>Up to two unknowns allowed to be imputed by kNN.</div>", unsafe_allow_html=True)

    # CHANGED: remove 3rd column (Tab 1 Settings) -> single input column
    left = st.container()

    with st.form("form_tab1"):
        # Input box
        with left:
            try:
                box = st.container(border=True)
            except TypeError:
                box = st.container()

            with box:
                c1, c2 = st.columns(2, gap="large")

                # Age
                with c1:
                    feature_header("Age (yr)")
                    # show NA toggle but keep disabled (Age required)
                    checkbox_unknown("tab1_na_age", default=False, disabled=True)
                    age_val = st.number_input(
                        label="",
                        min_value=1,
                        max_value=200,
                        value=int(st.session_state["Age (yr)"]),
                        step=1,
                        key="tab1_age",
                        label_visibility="collapsed",
                    )

                # Temperature
                with c2:
                    feature_header("Temperature (°C)")
                    temp_na = checkbox_unknown("tab1_temp_na", default=bool(st.session_state["temp_is_na"]))
                    if temp_na:
                        T_used = float(T0_meta)
                        _ = num_input_no_label(
                            "tab1_temp", -50.0, 60.0, float(T0_meta), STEPS["Temperature (°C)"], disabled=True
                        )
                    else:
                        T_used = float(
                            num_input_no_label(
                                "tab1_temp",
                                -50.0,
                                60.0,
                                float(st.session_state["Temperature (°C)"]),
                                STEPS["Temperature (°C)"],
                            )
                        )
                    # update shared
                    st.session_state["temp_is_na"] = bool(temp_na)
                    st.session_state["Temperature (°C)"] = float(T_used)

                # ML inputs (with NA flags)
                user_row_ml = {}

                # Soil_pH
                with c1:
                    feature_header("Soil_pH")
                    na = checkbox_unknown("tab1_na_Soil_pH", default=bool(st.session_state["na_Soil_pH"]))
                    st.session_state["na_Soil_pH"] = bool(na)
                    if na:
                        _ = num_input_no_label(
                            "tab1_Soil_pH", *RANGES["Soil_pH"], float(st.session_state["Soil_pH"]), STEPS["Soil_pH"], disabled=True
                        )
                        user_row_ml["Soil_pH"] = np.nan
                    else:
                        v = float(
                            num_input_no_label("tab1_Soil_pH", *RANGES["Soil_pH"], float(st.session_state["Soil_pH"]), STEPS["Soil_pH"])
                        )
                        st.session_state["Soil_pH"] = v
                        user_row_ml["Soil_pH"] = v

                # Chloride
                with c2:
                    feature_header("Chloride Content (mg/kg)")
                    na = checkbox_unknown("tab1_na_Chloride", default=bool(st.session_state["na_Chloride"]))
                    st.session_state["na_Chloride"] = bool(na)
                    if na:
                        _ = num_input_no_label(
                            "tab1_Chloride",
                            *RANGES["Chloride Content (mg/kg)"],
                            float(st.session_state["Chloride Content (mg/kg)"]),
                            STEPS["Chloride Content (mg/kg)"],
                            disabled=True,
                        )
                        user_row_ml["Chloride Content (mg/kg)"] = np.nan
                    else:
                        v = float(
                            num_input_no_label(
                                "tab1_Chloride",
                                *RANGES["Chloride Content (mg/kg)"],
                                float(st.session_state["Chloride Content (mg/kg)"]),
                                STEPS["Chloride Content (mg/kg)"],
                            )
                        )
                        st.session_state["Chloride Content (mg/kg)"] = v
                        user_row_ml["Chloride Content (mg/kg)"] = v

                # Resistivity
                with c1:
                    feature_header("Soil_Resistivity (Ω·cm)")
                    na = checkbox_unknown("tab1_na_Resistivity", default=bool(st.session_state["na_Resistivity"]))
                    st.session_state["na_Resistivity"] = bool(na)
                    if na:
                        _ = num_input_no_label(
                            "tab1_Resistivity",
                            *RANGES["Soil_Resistivity (Ω·cm)"],
                            float(st.session_state["Soil_Resistivity (Ω·cm)"]),
                            STEPS["Soil_Resistivity (Ω·cm)"],
                            disabled=True,
                        )
                        user_row_ml["Soil_Resistivity (Ω·cm)"] = np.nan
                    else:
                        v = float(
                            num_input_no_label(
                                "tab1_Resistivity",
                                *RANGES["Soil_Resistivity (Ω·cm)"],
                                float(st.session_state["Soil_Resistivity (Ω·cm)"]),
                                STEPS["Soil_Resistivity (Ω·cm)"],
                            )
                        )
                        st.session_state["Soil_Resistivity (Ω·cm)"] = v
                        user_row_ml["Soil_Resistivity (Ω·cm)"] = v

                # Sulphate
                with c2:
                    feature_header("Sulphate_Content (mg/kg)")
                    na = checkbox_unknown("tab1_na_Sulphate", default=bool(st.session_state["na_Sulphate"]))
                    st.session_state["na_Sulphate"] = bool(na)
                    if na:
                        _ = num_input_no_label(
                            "tab1_Sulphate",
                            *RANGES["Sulphate_Content (mg/kg)"],
                            float(st.session_state["Sulphate_Content (mg/kg)"]),
                            STEPS["Sulphate_Content (mg/kg)"],
                            disabled=True,
                        )
                        user_row_ml["Sulphate_Content (mg/kg)"] = np.nan
                    else:
                        v = float(
                            num_input_no_label(
                                "tab1_Sulphate",
                                *RANGES["Sulphate_Content (mg/kg)"],
                                float(st.session_state["Sulphate_Content (mg/kg)"]),
                                STEPS["Sulphate_Content (mg/kg)"],
                            )
                        )
                        st.session_state["Sulphate_Content (mg/kg)"] = v
                        user_row_ml["Sulphate_Content (mg/kg)"] = v

                # Moisture
                with c1:
                    feature_header("Moisture_Content (%)")
                    na = checkbox_unknown("tab1_na_Moisture", default=bool(st.session_state["na_Moisture"]))
                    st.session_state["na_Moisture"] = bool(na)
                    if na:
                        _ = num_input_no_label(
                            "tab1_Moisture",
                            *RANGES["Moisture_Content (%)"],
                            float(st.session_state["Moisture_Content (%)"]),
                            STEPS["Moisture_Content (%)"],
                            disabled=True,
                        )
                        user_row_ml["Moisture_Content (%)"] = np.nan
                    else:
                        v = float(
                            num_input_no_label(
                                "tab1_Moisture",
                                *RANGES["Moisture_Content (%)"],
                                float(st.session_state["Moisture_Content (%)"]),
                                STEPS["Moisture_Content (%)"],
                            )
                        )
                        st.session_state["Moisture_Content (%)"] = v
                        user_row_ml["Moisture_Content (%)"] = v

                # Soil type
                with c2:
                    feature_header("Soil Type (USCS)")
                    na = checkbox_unknown("tab1_na_SoilType", default=bool(st.session_state["na_SoilType"]))
                    st.session_state["na_SoilType"] = bool(na)
                    if na:
                        _ = select_input_no_label(
                            "tab1_SoilType",
                            SOIL_TYPES,
                            default_idx=SOIL_TYPES.index(st.session_state["Soil Type"]) if st.session_state["Soil Type"] in SOIL_TYPES else 1,
                            disabled=True,
                        )
                        user_row_ml["Soil Type"] = np.nan
                    else:
                        v = select_input_no_label(
                            "tab1_SoilType",
                            SOIL_TYPES,
                            default_idx=SOIL_TYPES.index(st.session_state["Soil Type"]) if st.session_state["Soil Type"] in SOIL_TYPES else 1,
                        )
                        st.session_state["Soil Type"] = v
                        user_row_ml["Soil Type"] = v

                # Water table
                with c1:
                    feature_header("Location wrt Water Table")
                    na = checkbox_unknown("tab1_na_WT", default=bool(st.session_state["na_WT"]))
                    st.session_state["na_WT"] = bool(na)
                    if na:
                        _ = select_input_no_label(
                            "tab1_WT",
                            WATER_TABLE,
                            default_idx=WATER_TABLE.index(st.session_state["Location wrt Water Table"])
                            if st.session_state["Location wrt Water Table"] in WATER_TABLE
                            else 0,
                            disabled=True,
                        )
                        user_row_ml["Location wrt Water Table"] = np.nan
                    else:
                        v = select_input_no_label(
                            "tab1_WT",
                            WATER_TABLE,
                            default_idx=WATER_TABLE.index(st.session_state["Location wrt Water Table"])
                            if st.session_state["Location wrt Water Table"] in WATER_TABLE
                            else 0,
                        )
                        st.session_state["Location wrt Water Table"] = v
                        user_row_ml["Location wrt Water Table"] = v

                # Foreign inclusion type
                with c2:
                    feature_header("Foreign_Inclusion_Type")
                    na = checkbox_unknown("tab1_na_Foreign", default=bool(st.session_state["na_Foreign"]))
                    st.session_state["na_Foreign"] = bool(na)
                    if na:
                        _ = select_input_no_label(
                            "tab1_Foreign",
                            FOREIGN_INCL,
                            default_idx=FOREIGN_INCL.index(st.session_state["Foreign_Inclusion_Type"])
                            if st.session_state["Foreign_Inclusion_Type"] in FOREIGN_INCL
                            else 0,
                            disabled=True,
                        )
                        user_row_ml["Foreign_Inclusion_Type"] = np.nan
                    else:
                        v = select_input_no_label(
                            "tab1_Foreign",
                            FOREIGN_INCL,
                            default_idx=FOREIGN_INCL.index(st.session_state["Foreign_Inclusion_Type"])
                            if st.session_state["Foreign_Inclusion_Type"] in FOREIGN_INCL
                            else 0,
                        )
                        st.session_state["Foreign_Inclusion_Type"] = v
                        user_row_ml["Foreign_Inclusion_Type"] = v

                # Fill
                with c1:
                    feature_header("Is_Fill_Material")
                    na = checkbox_unknown("tab1_na_Fill", default=bool(st.session_state["na_Fill"]))
                    st.session_state["na_Fill"] = bool(na)
                    if na:
                        _ = select_input_no_label("tab1_Fill", FILL_MATERIAL, default_idx=0, disabled=True)
                        user_row_ml["Is_Fill_Material"] = np.nan
                    else:
                        v = select_input_no_label(
                            "tab1_Fill",
                            FILL_MATERIAL,
                            default_idx=FILL_MATERIAL.index(st.session_state["Is_Fill_Material"])
                            if st.session_state["Is_Fill_Material"] in FILL_MATERIAL
                            else 0,
                        )
                        st.session_state["Is_Fill_Material"] = int(v)
                        user_row_ml["Is_Fill_Material"] = int(v)

        run1 = st.form_submit_button("Run standards + ML (mean)")

    # ------- OUTPUT TAB 1 -------
    if run1:
        # update shared age
        st.session_state["Age (yr)"] = int(age_val)
        age_val = int(age_val)

        # check missing limit for ML
        miss = count_missing_ml_features(user_row_ml, expected_cols)
        if miss > 2:
            st.error(f"Too many unknown ML inputs: {miss}. Maximum allowed is 2.")
            st.stop()

        # Build ML input df in correct order
        X_in = pd.DataFrame([{c: user_row_ml.get(c, np.nan) for c in expected_cols}])

        # Predict k distribution
        try:
            X_tr = prep.transform(X_in)
            mu_k_arr, sd_k_arr = gpr.predict(np.asarray(X_tr, float), return_std=True)
            mu_k = float(mu_k_arr[0])
            sd_k = float(max(sd_k_arr[0], EPS))
        except Exception as e:
            st.error(f"ML prediction failed: {e}")
            st.stop()

        # MC mean-only TL at selected age (fixed MC settings)
        n_bounds = (0.40, 0.70)
        b_bounds = (0.020, 0.040)
        Ns_fixed = 5000
        shared_fixed = False

        mc_single = mc_TL_from_k(
            mu_k=mu_k,
            sd_k=sd_k,
            ages=[age_val],
            T_used=float(T_used),
            T0=float(T0_meta),
            mu_n=float(mu_n_meta),
            mu_beta=float(mu_beta_meta),
            n_bounds=n_bounds,
            beta_bounds=b_bounds,
            Ns=Ns_fixed,
            seed=42,
            shared_n_beta=shared_fixed,
        ).iloc[0]
        ml_TL_mean = float(mc_single["Mean_TL (mm)"])

        # Standards inputs row dict (derive foreign flag from foreign type; no separate toggle)
        foreign_type_for_flag = user_row_ml.get("Foreign_Inclusion_Type", st.session_state["Foreign_Inclusion_Type"])
        foreign_flag = derive_foreign_flag(foreign_type_for_flag)

        row_std = {
            COL_AGE: age_val,
            COL_PH: user_row_ml.get("Soil_pH", np.nan),
            COL_CL: user_row_ml.get("Chloride Content (mg/kg)", np.nan),
            COL_SO4: user_row_ml.get("Sulphate_Content (mg/kg)", np.nan),
            COL_RHO: user_row_ml.get("Soil_Resistivity (Ω·cm)", np.nan),
            COL_SOIL: user_row_ml.get("Soil Type", np.nan),
            COL_LOC: user_row_ml.get("Location wrt Water Table", np.nan),
            COL_FILL: user_row_ml.get("Is_Fill_Material", np.nan),
            COL_FOREIGN: foreign_flag,
            COL_FTYPE: foreign_type_for_flag,
        }

        rates, notes = compute_all_standards(row_std)

        # Build output table: thickness loss over age (rate*age). If rate NaN -> show NA and note.
        rows = []
        for key in STANDARD_KEYS:
            rate = rates.get(key, np.nan)
            loss = np.nan if np.isnan(rate) else float(rate) * float(age_val)
            rows.append(
                {
                    "Method": DISPLAY_NAMES.get(key, key),
                    f"Thickness Loss at {age_val} yr (mm)": loss,
                    "Note": notes.get(key, ""),
                }
            )

        # Add ML as a "method" row (thickness loss only)
        rows.append(
            {
                "Method": "Machine Learning (Mean Value)",
                f"Thickness Loss at {age_val} yr (mm)": ml_TL_mean,
                "Note": "",
            }
        )

        df_out = pd.DataFrame(rows)

        st.markdown("<div class='sectiontitle'>Output</div>", unsafe_allow_html=True)

        out_left, out_right = st.columns([1.15, 1.0], gap="large")

        with out_left:
            # CHANGED: directly show table (no ML k lines, no ML TL text)
            st.dataframe(
                df_out.style.format({f"Thickness Loss at {age_val} yr (mm)": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download table (CSV)",
                data=csv_bytes,
                file_name=f"standards_ml_thickness_loss_age_{age_val}.csv",
                mime="text/csv",
            )

        with out_right:
            # Bar plot: thickness loss only (standards + ML)
            plot_df = df_out.copy()
            plot_df["Short"] = [
                SHORT_NAMES.get("CSA_S6", "CSA"),
                SHORT_NAMES.get("Eurocode", "EN"),
                SHORT_NAMES.get("AS2159", "AS"),
                SHORT_NAMES.get("NZS", "NZS"),
                SHORT_NAMES.get("WSDOT", "WSDOT"),
                SHORT_NAMES.get("FDOT", "FDOT"),
                SHORT_NAMES.get("Japan", "OCDI"),
                SHORT_NAMES.get("China", "JTG"),
                SHORT_NAMES.get("DIN", "DIN"),
                SHORT_NAMES.get("Caltrans", "Caltrans"),
                "ML",
            ]

            colors = [
                STANDARD_COLORS.get("CSA_S6"),
                STANDARD_COLORS.get("Eurocode"),
                STANDARD_COLORS.get("AS2159"),
                STANDARD_COLORS.get("NZS"),
                STANDARD_COLORS.get("WSDOT"),
                STANDARD_COLORS.get("FDOT"),
                STANDARD_COLORS.get("Japan"),
                STANDARD_COLORS.get("China"),
                STANDARD_COLORS.get("DIN"),
                STANDARD_COLORS.get("Caltrans"),
                "#444444",
            ]

            yvals = plot_df[f"Thickness Loss at {age_val} yr (mm)"].values.astype(float)
            xlabels = plot_df["Short"].values

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(np.arange(len(xlabels)), yvals, color=colors, edgecolor="black")
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(xlabels, fontsize=14)
            ax.set_ylabel(f"Thickness Loss at {age_val} yr (mm)", fontsize=16)
            ax.yaxis.set_major_formatter(FMT)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            st.download_button(
                "Download bar plot (PNG)",
                data=fig_to_png_bytes(fig),
                file_name=f"thickness_loss_bar_age_{age_val}.png",
                mime="image/png",
            )


# ============================================================
# TAB 2: ML + MONTE CARLO (DETAILED)
# ============================================================
with tab2:
    st.markdown("<div class='sectiontitle'>Input Parameters</div>", unsafe_allow_html=True)
    st.markdown("<div class='sectionnote'>Inputs are prefilled from Tab 1. Up to two unknowns allowed to be imputed by kNN.</div>", unsafe_allow_html=True)

    left, right = st.columns([2.4, 1.1], gap="large")

    with st.form("form_tab2"):
        with left:
            try:
                box = st.container(border=True)
            except TypeError:
                box = st.container()

            with box:
                c1, c2 = st.columns(2, gap="large")

                # Age
                with c1:
                    feature_header("Age (yr)")
                    checkbox_unknown("tab2_na_age", default=False, disabled=True)
                    age_now = st.number_input(
                        label="",
                        min_value=1,
                        max_value=200,
                        value=int(st.session_state["Age (yr)"]),
                        step=1,
                        key="tab2_age",
                        label_visibility="collapsed",
                    )

                # Temperature
                with c2:
                    feature_header("Temperature (°C)")
                    temp_na = checkbox_unknown("tab2_temp_na", default=bool(st.session_state["temp_is_na"]))
                    if temp_na:
                        T_used = float(T0_meta)
                        _ = num_input_no_label("tab2_temp", -50.0, 60.0, float(T0_meta), STEPS["Temperature (°C)"], disabled=True)
                    else:
                        T_used = float(num_input_no_label("tab2_temp", -50.0, 60.0, float(st.session_state["Temperature (°C)"]), STEPS["Temperature (°C)"]))
                    st.session_state["temp_is_na"] = bool(temp_na)
                    st.session_state["Temperature (°C)"] = float(T_used)

                # ML row
                user_row = {}

                # Soil_pH
                with c1:
                    feature_header("Soil_pH")
                    na = checkbox_unknown("tab2_na_Soil_pH", default=bool(st.session_state["na_Soil_pH"]))
                    st.session_state["na_Soil_pH"] = bool(na)
                    if na:
                        _ = num_input_no_label("tab2_Soil_pH", *RANGES["Soil_pH"], float(st.session_state["Soil_pH"]), STEPS["Soil_pH"], disabled=True)
                        user_row["Soil_pH"] = np.nan
                    else:
                        v = float(num_input_no_label("tab2_Soil_pH", *RANGES["Soil_pH"], float(st.session_state["Soil_pH"]), STEPS["Soil_pH"]))
                        st.session_state["Soil_pH"] = v
                        user_row["Soil_pH"] = v

                # Chloride
                with c2:
                    feature_header("Chloride Content (mg/kg)")
                    na = checkbox_unknown("tab2_na_Chloride", default=bool(st.session_state["na_Chloride"]))
                    st.session_state["na_Chloride"] = bool(na)
                    if na:
                        _ = num_input_no_label("tab2_Chloride", *RANGES["Chloride Content (mg/kg)"], float(st.session_state["Chloride Content (mg/kg)"]), STEPS["Chloride Content (mg/kg)"], disabled=True)
                        user_row["Chloride Content (mg/kg)"] = np.nan
                    else:
                        v = float(num_input_no_label("tab2_Chloride", *RANGES["Chloride Content (mg/kg)"], float(st.session_state["Chloride Content (mg/kg)"]), STEPS["Chloride Content (mg/kg)"]))
                        st.session_state["Chloride Content (mg/kg)"] = v
                        user_row["Chloride Content (mg/kg)"] = v

                # Resistivity
                with c1:
                    feature_header("Soil_Resistivity (Ω·cm)")
                    na = checkbox_unknown("tab2_na_Resistivity", default=bool(st.session_state["na_Resistivity"]))
                    st.session_state["na_Resistivity"] = bool(na)
                    if na:
                        _ = num_input_no_label("tab2_Resistivity", *RANGES["Soil_Resistivity (Ω·cm)"], float(st.session_state["Soil_Resistivity (Ω·cm)"]), STEPS["Soil_Resistivity (Ω·cm)"], disabled=True)
                        user_row["Soil_Resistivity (Ω·cm)"] = np.nan
                    else:
                        v = float(num_input_no_label("tab2_Resistivity", *RANGES["Soil_Resistivity (Ω·cm)"], float(st.session_state["Soil_Resistivity (Ω·cm)"]), STEPS["Soil_Resistivity (Ω·cm)"]))
                        st.session_state["Soil_Resistivity (Ω·cm)"] = v
                        user_row["Soil_Resistivity (Ω·cm)"] = v

                # Sulphate
                with c2:
                    feature_header("Sulphate_Content (mg/kg)")
                    na = checkbox_unknown("tab2_na_Sulphate", default=bool(st.session_state["na_Sulphate"]))
                    st.session_state["na_Sulphate"] = bool(na)
                    if na:
                        _ = num_input_no_label("tab2_Sulphate", *RANGES["Sulphate_Content (mg/kg)"], float(st.session_state["Sulphate_Content (mg/kg)"]), STEPS["Sulphate_Content (mg/kg)"], disabled=True)
                        user_row["Sulphate_Content (mg/kg)"] = np.nan
                    else:
                        v = float(num_input_no_label("tab2_Sulphate", *RANGES["Sulphate_Content (mg/kg)"], float(st.session_state["Sulphate_Content (mg/kg)"]), STEPS["Sulphate_Content (mg/kg)"]))
                        st.session_state["Sulphate_Content (mg/kg)"] = v
                        user_row["Sulphate_Content (mg/kg)"] = v

                # Moisture
                with c1:
                    feature_header("Moisture_Content (%)")
                    na = checkbox_unknown("tab2_na_Moisture", default=bool(st.session_state["na_Moisture"]))
                    st.session_state["na_Moisture"] = bool(na)
                    if na:
                        _ = num_input_no_label("tab2_Moisture", *RANGES["Moisture_Content (%)"], float(st.session_state["Moisture_Content (%)"]), STEPS["Moisture_Content (%)"], disabled=True)
                        user_row["Moisture_Content (%)"] = np.nan
                    else:
                        v = float(num_input_no_label("tab2_Moisture", *RANGES["Moisture_Content (%)"], float(st.session_state["Moisture_Content (%)"]), STEPS["Moisture_Content (%)"]))
                        st.session_state["Moisture_Content (%)"] = v
                        user_row["Moisture_Content (%)"] = v

                # Soil type
                with c2:
                    feature_header("Soil Type (USCS)")
                    na = checkbox_unknown("tab2_na_SoilType", default=bool(st.session_state["na_SoilType"]))
                    st.session_state["na_SoilType"] = bool(na)
                    if na:
                        _ = select_input_no_label("tab2_SoilType", SOIL_TYPES, default_idx=SOIL_TYPES.index(st.session_state["Soil Type"]) if st.session_state["Soil Type"] in SOIL_TYPES else 1, disabled=True)
                        user_row["Soil Type"] = np.nan
                    else:
                        v = select_input_no_label("tab2_SoilType", SOIL_TYPES, default_idx=SOIL_TYPES.index(st.session_state["Soil Type"]) if st.session_state["Soil Type"] in SOIL_TYPES else 1)
                        st.session_state["Soil Type"] = v
                        user_row["Soil Type"] = v

                # Water table
                with c1:
                    feature_header("Location wrt Water Table")
                    na = checkbox_unknown("tab2_na_WT", default=bool(st.session_state["na_WT"]))
                    st.session_state["na_WT"] = bool(na)
                    if na:
                        _ = select_input_no_label("tab2_WT", WATER_TABLE, default_idx=WATER_TABLE.index(st.session_state["Location wrt Water Table"]) if st.session_state["Location wrt Water Table"] in WATER_TABLE else 0, disabled=True)
                        user_row["Location wrt Water Table"] = np.nan
                    else:
                        v = select_input_no_label("tab2_WT", WATER_TABLE, default_idx=WATER_TABLE.index(st.session_state["Location wrt Water Table"]) if st.session_state["Location wrt Water Table"] in WATER_TABLE else 0)
                        st.session_state["Location wrt Water Table"] = v
                        user_row["Location wrt Water Table"] = v

                # Foreign inclusion type
                with c2:
                    feature_header("Foreign_Inclusion_Type")
                    na = checkbox_unknown("tab2_na_Foreign", default=bool(st.session_state["na_Foreign"]))
                    st.session_state["na_Foreign"] = bool(na)
                    if na:
                        _ = select_input_no_label("tab2_Foreign", FOREIGN_INCL, default_idx=FOREIGN_INCL.index(st.session_state["Foreign_Inclusion_Type"]) if st.session_state["Foreign_Inclusion_Type"] in FOREIGN_INCL else 0, disabled=True)
                        user_row["Foreign_Inclusion_Type"] = np.nan
                    else:
                        v = select_input_no_label("tab2_Foreign", FOREIGN_INCL, default_idx=FOREIGN_INCL.index(st.session_state["Foreign_Inclusion_Type"]) if st.session_state["Foreign_Inclusion_Type"] in FOREIGN_INCL else 0)
                        st.session_state["Foreign_Inclusion_Type"] = v
                        user_row["Foreign_Inclusion_Type"] = v

                # Fill
                with c1:
                    feature_header("Is_Fill_Material")
                    na = checkbox_unknown("tab2_na_Fill", default=bool(st.session_state["na_Fill"]))
                    st.session_state["na_Fill"] = bool(na)
                    if na:
                        _ = select_input_no_label("tab2_Fill", FILL_MATERIAL, default_idx=0, disabled=True)
                        user_row["Is_Fill_Material"] = np.nan
                    else:
                        v = select_input_no_label("tab2_Fill", FILL_MATERIAL, default_idx=FILL_MATERIAL.index(st.session_state["Is_Fill_Material"]) if st.session_state["Is_Fill_Material"] in FILL_MATERIAL else 0)
                        st.session_state["Is_Fill_Material"] = int(v)
                        user_row["Is_Fill_Material"] = int(v)

        # MC settings (editable)
        with right:
            try:
                boxr = st.container(border=True)
            except TypeError:
                boxr = st.container()
            with boxr:
                st.markdown(f"<div class='sectiontitle' style='margin-top:0;'>Monte Carlo Settings</div>", unsafe_allow_html=True)
                st.markdown("<div class='sectionnote' style='margin-bottom:10px;'> </div>", unsafe_allow_html=True)

                Ns = st.number_input("Sample size (Ns)", min_value=1000, max_value=50000, value=MC_NS_DEFAULT, step=1000, key="tab2_Ns")
                nL = st.number_input("n lower bound", value=0.40, step=0.01, format="%.2f", key="tab2_nL")
                nU = st.number_input("n upper bound", value=0.70, step=0.01, format="%.2f", key="tab2_nU")
                bL = st.number_input("β lower bound", value=0.020, step=0.001, format="%.3f", key="tab2_bL")
                bU = st.number_input("β upper bound", value=0.040, step=0.001, format="%.3f", key="tab2_bU")
                shared = st.checkbox("Shared n, β across samples", value=False, key="tab2_shared")

        run2 = st.form_submit_button("Run predictions + Monte Carlo")

    # ------- OUTPUT TAB 2 -------
    if run2:
        # update shared age
        st.session_state["Age (yr)"] = int(age_now)
        age_now = int(age_now)

        miss = count_missing_ml_features(user_row, expected_cols)
        if miss > 2:
            st.error(f"Too many unknown ML inputs: {miss}. Maximum allowed is 2.")
            st.stop()

        X_in = pd.DataFrame([{c: user_row.get(c, np.nan) for c in expected_cols}])

        try:
            X_tr = prep.transform(X_in)
            mu_k_arr, sd_k_arr = gpr.predict(np.asarray(X_tr, float), return_std=True)
            mu_k = float(mu_k_arr[0])
            sd_k = float(max(sd_k_arr[0], EPS))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Single age
        single = mc_TL_from_k(
            mu_k=mu_k,
            sd_k=sd_k,
            ages=[age_now],
            T_used=float(T_used),
            T0=float(T0_meta),
            mu_n=float(mu_n_meta),
            mu_beta=float(mu_beta_meta),
            n_bounds=(float(nL), float(nU)),
            beta_bounds=(float(bL), float(bU)),
            Ns=int(Ns),
            seed=42,
            shared_n_beta=bool(shared),
        ).iloc[0]

        mean_TL = float(single["Mean_TL (mm)"])
        sd_TL = float(single["TL_sd (mm)"])

        # Horizon
        horizon_df = mc_TL_from_k(
            mu_k=mu_k,
            sd_k=sd_k,
            ages=AGES_HORIZON,
            T_used=float(T_used),
            T0=float(T0_meta),
            mu_n=float(mu_n_meta),
            mu_beta=float(mu_beta_meta),
            n_bounds=(float(nL), float(nU)),
            beta_bounds=(float(bL), float(bU)),
            Ns=int(Ns),
            seed=123,
            shared_n_beta=bool(shared),
        )

        out_tbl = pd.DataFrame(
            {
                "Age": horizon_df["Age"].astype(int),
                "Mean_TL (mm)": horizon_df["Mean_TL (mm)"].round(3),
                "TL_sd (mm)": horizon_df["TL_sd (mm)"].round(3),
                "TL (68% CI)": [fmt_ci(a, b, 3) for a, b in zip(horizon_df["TL_lo68 (mm)"], horizon_df["TL_hi68 (mm)"])],
                "TL (95% CI)": [fmt_ci(a, b, 3) for a, b in zip(horizon_df["TL_lo95 (mm)"], horizon_df["TL_hi95 (mm)"])],
            }
        )

        st.markdown("<div class='sectiontitle'>Output</div>", unsafe_allow_html=True)

        out_left, out_right = st.columns([1.2, 1.0], gap="large")

        with out_left:
            st.markdown("<div class='outline'><b>Predicted k from ML</b></div>", unsafe_allow_html=True)
            st.latex(rf"\mu_k={mu_k:.6f}\qquad \sigma_k={sd_k:.6f}")

            st.markdown(f"<div class='outline'><b>Thickness loss at Input Age ({age_now} years)</b></div>", unsafe_allow_html=True)
            st.latex(rf"\text{{68\% CI: }} {mean_TL:.3f}\pm{sd_TL:.3f}\qquad \text{{95\% CI: }} {mean_TL:.3f}\pm{2.0*sd_TL:.3f}")

            st.markdown("<div class='sectiontitle'>Thickness Loss across Time</div>", unsafe_allow_html=True)
            st.dataframe(out_tbl, use_container_width=True, hide_index=True)

            csv_bytes = out_tbl.to_csv(index=False).encode("utf-8")
            st.download_button("Download TL table (CSV)", data=csv_bytes, file_name="thickness_loss_table.csv", mime="text/csv")

        with out_right:
            ages = horizon_df["Age"].values
            mean = horizon_df["Mean_TL (mm)"].values
            lo68 = horizon_df["TL_lo68 (mm)"].values
            hi68 = horizon_df["TL_hi68 (mm)"].values
            lo95 = horizon_df["TL_lo95 (mm)"].values
            hi95 = horizon_df["TL_hi95 (mm)"].values

            fig = plt.figure(figsize=(10, 6))
            plt.plot(ages, mean, linewidth=2, label="Mean TL")
            plt.fill_between(ages, lo95, hi95, alpha=0.20, label="95% CI")
            plt.fill_between(ages, lo68, hi68, alpha=0.35, label="68% CI")
            plt.xlabel("Age (year)")
            plt.ylabel("Thickness Loss (mm)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)

            st.download_button(
                "Download plot (PNG)",
                data=fig_to_png_bytes(fig),
                file_name="thickness_loss_plot.png",
                mime="image/png",
            )

# streamlit_app.py
import re, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pulp as pl

# sklearn imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ---- Defaults (tek kaynak) ----
PER_RECYCLE_DEFAULT = 120_000.0
PER_TREAT_DEFAULT   = 350_000.0
DEFAULT_START_YEAR  = 2020
DEFAULT_END_YEAR    = 2025

# ---- Preset yardımcıları ----
def _set_if_missing(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset_to_session(name: str):
    presets = {
        "Baseline": {
            "sp_recycle_target": 0.60, "sp_min_treat": 0.15, "sp_max_landfill": 0.60,
            "sp_target_band": 0.02, "sp_carbon_price": 50.0, "sp_growth": 1.00,
            "sp_uncertainty": 0.00, "sp_discount": 0.08,
            "cap_recycle_max": 0.85, "cap_treat_max": 0.35, "cap_landfill_min": 0.02,
            "pen_overshoot": 600.0, "pen_landfill_under": 150.0,
            "per_recycle": 120_000.0, "per_treat": 350_000.0,
            "max_new_R": 10, "max_new_T": 22,
            "lead_recycle": 0, "lead_treatment": 1,
        },
        "Budget": {
            "sp_recycle_target": 0.60, "sp_min_treat": 0.15, "sp_max_landfill": 0.60,
            "sp_target_band": 0.01, "sp_carbon_price": 0.0, "sp_growth": 1.00,
            "sp_uncertainty": 0.00, "sp_discount": 0.08,
            "cap_recycle_max": 0.85, "cap_treat_max": 0.35, "cap_landfill_min": 0.02,
            "pen_overshoot": 600.0, "pen_landfill_under": 150.0,
            "per_recycle": 120_000.0, "per_treat": 350_000.0,
            "max_new_R": 6, "max_new_T": 10,
            "lead_recycle": 1, "lead_treatment": 1,
        },
        "Green": {
            "sp_recycle_target": 0.60, "sp_min_treat": 0.15, "sp_max_landfill": 0.50,
            "sp_target_band": 0.02, "sp_carbon_price": 100.0, "sp_growth": 1.00,
            "sp_uncertainty": 0.00, "sp_discount": 0.08,
            "cap_recycle_max": 0.90, "cap_treat_max": 0.38, "cap_landfill_min": 0.01,
            "pen_overshoot": 600.0, "pen_landfill_under": 150.0,
            "per_recycle": 120_000.0, "per_treat": 350_000.0,
            "max_new_R": 15, "max_new_T": 25,
            "lead_recycle": 0, "lead_treatment": 1,
        },
    }
    p = presets.get(name)
    if not p:
        return
    for k, v in p.items():
        st.session_state[k] = v


st.set_page_config(page_title="MSW DSS — Türkiye (RF + DSS v3)", page_icon="♻️", layout="wide")

# =========================
# Utilities & Defaults
# =========================
def _clean_decimal_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c):
        s = c.lower()
        s = (s.replace("ı","i").replace("ğ","g").replace("ü","u")
               .replace("ş","s").replace("ö","o").replace("ç","c"))
        s = re.sub(r"\s+", " ", s).strip()
        return s
    return df.rename(columns={c: norm(c) for c in df.columns})

def _has_thousand_word(colname: str) -> bool:
    c = colname.lower()
    return ("thousand" in c) or ("bin" in c)

# Örnek ulusal veri (fallback)
SAMPLE_DF = pd.DataFrame({
    "region": ["National"]*3,
    "year":   [2020, 2021, 2022],
    "waste_collected_ton_per_year": [30_000_000, 29_455_750, 30_283_760],
    "waste_generated_ton_per_year": [32_000_000, 33_940_700, 31_797_940],
})

# --- Repo içi varsayılan dosyalar ---
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MAIN_PATH = DATA_DIR / "belediye_atik (2).xlsx"
DEFAULT_FAC_PATH  = DATA_DIR / "tesis.xlsx"

# -------------------------------------------------------------------
# A) ULUSAL VERIYI TOLERANSLI OKU (yil/region/kolon ad eşleştirme)
# -------------------------------------------------------------------
MAIN_SYNONYMS = {
    # 'Date' eklendi
    "year":   [r"\byear\b", r"\byil\b", r"\byıl\b", r"\bdate\b"],
    "region": [r"\bregion\b", r"\bbolge\b", r"\bbolge\b", r"\bill\b", r"\bsehir\b"],
    "waste_collected_ton_per_year": [
        r"waste.*collec", r"toplanan.*atik", r"toplanan_atik", r"collected.*ton"
    ],
    "waste_generated_ton_per_year": [
        r"waste.*generat", r"uretilen.*atik", r"üretilen.*atik", r"generated.*ton"
    ],
    "landfill_capacity_ton": [r"landfill.*cap", r"depolama.*kapasite"],
    "recycle_capacity_ton":  [r"recyc.*cap", r"geri donus.*kapasite", r"geri donusum.*kapasite"],
    "treatment_capacity_ton":[r"treat.*cap", r"ar(i|ı)tma.*kapasite", r"bertaraf.*kapasite"]
}

def _find_first_match(cols, patterns):
    for pat in patterns:
        rx = re.compile(pat)
        for c in cols:
            if rx.search(c):
                return c
    return None

def load_main_table(file_or_path) -> pd.DataFrame:
    # Dosyayı oku (ilk sheet yeterli)
    name = getattr(file_or_path, "name", None)
    ext = (name or str(file_or_path)).lower().split(".")[-1]
    if ext in ("xlsx","xls"):
        raw = pd.read_excel(file_or_path)
    elif ext == "csv":
        raw = pd.read_csv(file_or_path)
    else:
        st.error("Ulusal veri: .xlsx/.xls/.csv yükleyin.")
        st.stop()

    raw = normalize_cols(raw)

    # Yıl kolonu: önce eşle, olmazsa 1. sütun 1900–2100 aralığı
    year_col = _find_first_match(raw.columns, MAIN_SYNONYMS["year"])
    if year_col is None:
        first = raw.columns[0]
        cand = pd.to_numeric(raw[first], errors="coerce")
        if cand.between(1900, 2100).any():
            year_col = first
    if year_col is None:
        st.error("Ulusal veri: Yıl kolonu (year/yıl/yil/date) bulunamadı.")
        st.stop()

    # 4 haneli yıl çek + aralık filtresi
    y = pd.to_numeric(raw[year_col].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    y = y.where(y.between(1900, 2100))

    df = raw.copy()
    df["year"] = y
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].round().astype(int)

    # Region yoksa National
    reg_col = _find_first_match(df.columns, MAIN_SYNONYMS["region"])
    if reg_col and reg_col != "region":
        df.rename(columns={reg_col: "region"}, inplace=True)
    if "region" not in df.columns:
        df["region"] = "National"

    # İçerik kolonlarını eşleştir + "thousand/bin" ise ×1000
    out = df[["year","region"]].copy()
    for std_key in [
        "waste_collected_ton_per_year",
        "waste_generated_ton_per_year",
        "landfill_capacity_ton",
        "recycle_capacity_ton",
        "treatment_capacity_ton",
    ]:
        src = _find_first_match(df.columns, MAIN_SYNONYMS.get(std_key, []))
        if src:
            series = _clean_decimal_series(df[src])
            if _has_thousand_word(src):
                series = series * 1000.0
            out[std_key] = series

    if not (("waste_collected_ton_per_year" in out.columns) or
            ("waste_generated_ton_per_year" in out.columns)):
        st.error("Ulusal veri: Toplanan/Üretilen atık kolonlarından en az biri bulunmalı.")
        st.stop()

    return out.sort_values(["year","region"]).reset_index(drop=True)

# -------------------------------------------------------------------
# B) TESIS TABLOSU (RF IMPUTE)
# -------------------------------------------------------------------
def load_facility_table(file_or_path):
    if file_or_path is None:
        return None
    name = getattr(file_or_path, "name", None)
    ext = (name or str(file_or_path)).lower().split(".")[-1]
    if ext in ("xlsx","xls"):
        raw = pd.read_excel(file_or_path)
    elif ext == "csv":
        raw = pd.read_csv(file_or_path)
    else:
        st.error("Tesis tablosu: .xlsx/.xls/.csv yükleyin.")
        st.stop()

    raw = normalize_cols(raw)

    if "year" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "year"})
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")

    colmap = {
        "kompost tesisi (tesis sayisi)": "compost_facilities",
        "beraber yakma (ko-insinerasyon) tesisi (tesis sayisi)": "coincin_facilities",
        "diger geri kazanim tesisleri (tesis sayisi)": "recovery_facilities",
        "kompost tesisi (islenen atik miktari)": "compost_processed_ton",
        "beraber yakma (ko-insinerasyon) tesisi (islenen atik miktari)": "coincin_processed_ton",
        "diger geri kazanim tesisleri (islenen atik miktari)": "recovery_processed_ton",
        "geri donusum oranlari (%)": "recycle_rate_pct",
        "eu geri donusum hedefleri (%)": "eu_recycle_target_pct",
    }
    for k,v in colmap.items():
        if k in raw.columns: raw.rename(columns={k:v}, inplace=True)

    keep = ["year"] + list(colmap.values())
    dfF = raw[[c for c in keep if c in raw.columns]].copy()

    for c in dfF.columns:
        if c == "year": continue
        dfF[c] = _clean_decimal_series(dfF[c])

    dfF = dfF.sort_values("year")
    num_cols = [c for c in dfF.columns if c!="year"]
    for c in num_cols:
        dfF[f"{c}_lag1"] = dfF[c].shift(1)

    rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    imputer = IterativeImputer(estimator=rf, random_state=42, max_iter=20)
    imputed = dfF.copy()
    imp_cols = num_cols + [c for c in dfF.columns if c.endswith("_lag1")]
    imputed[imp_cols] = imputer.fit_transform(imputed[imp_cols])

    for c in ["compost_facilities","coincin_facilities","recovery_facilities"]:
        if c in imputed.columns:
            imputed[c] = np.clip(np.round(imputed[c]), 0, None).astype(int)

    imputed.drop(columns=[c for c in imputed.columns if c.endswith("_lag1")], inplace=True, errors="ignore")
    return imputed

# -----------------------
# C) Kapasite çıkarımı
# -----------------------
def capacities_by_year_from_facility(dfF):
    dfF = dfF.sort_values("year").reset_index(drop=True)

    def safe_ratio(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
        return num / den

    rec_perfac = safe_ratio(dfF.get("recovery_processed_ton"), dfF.get("recovery_facilities"))
    trt_processed = (pd.to_numeric(dfF.get("compost_processed_ton"), errors="coerce").fillna(0) +
                     pd.to_numeric(dfF.get("coincin_processed_ton"), errors="coerce").fillna(0))
    trt_facilities = (pd.to_numeric(dfF.get("compost_facilities"), errors="coerce").fillna(0) +
                      pd.to_numeric(dfF.get("coincin_facilities"), errors="coerce").fillna(0))
    trt_perfac = safe_ratio(trt_processed, trt_facilities)

    def clip_1_99(s):
        s = s.dropna()
        if s.empty: return s
        q1,q99 = np.nanpercentile(s, [1,99])
        return s.clip(q1,q99)

    rec_perfac = clip_1_99(rec_perfac).reindex(dfF.index).interpolate("linear", limit_direction="both")
    trt_perfac = clip_1_99(trt_perfac).reindex(dfF.index).interpolate("linear", limit_direction="both")

    base_caps = {}
    for i,row in dfF.iterrows():
        y = int(row["year"])
        n_rec = float(row.get("recovery_facilities", 0) or 0)
        n_trt = float((row.get("compost_facilities",0) or 0) + (row.get("coincin_facilities",0) or 0))
        base_caps[y] = {
            "recycle":   float(n_rec * (rec_perfac.iloc[i] if pd.notna(rec_perfac.iloc[i]) else 0.0)),
            "treatment": float(n_trt * (trt_perfac.iloc[i] if pd.notna(trt_perfac.iloc[i]) else 0.0))
        }

    per_fac = {
        "recycle":   float(np.nanmedian(rec_perfac)),
        "treatment": float(np.nanmedian(trt_perfac))
    }
    return base_caps, per_fac

def build_target_series_from_facility(dfF):
    if dfF is None or "eu_recycle_target_pct" not in dfF.columns:
        return None
    t = dfF[["year","eu_recycle_target_pct"]].dropna()
    if t.empty: return None
    t["eu_recycle_target_pct"] = pd.to_numeric(t["eu_recycle_target_pct"], errors="coerce")/100.0
    t = t.set_index("year")["eu_recycle_target_pct"].sort_index()
    t = t.interpolate("linear", limit_direction="both")
    return {int(k): float(v) for k,v in t.items()}

# ------------------------------------------------------------
# D) DSS v3
# ------------------------------------------------------------
def solve_with_dynamic_v3(df_main, scenario_name, sp, YEARS,
                          target_by_year=None,
                          per_fac_override=None,
                          base_caps_by_year=None,
                          landfill_share_for_base=0.65,
                          ramp_limits=True,
                          max_new_per_year={"recycle":10, "treatment":22, "landfill":0},
                          share_caps={"recycle_max":0.85, "treat_max":0.35, "landfill_min":0.02},
                          target_band_up=0.02,
                          extra_penalties={"recycle_overshoot":600, "treat_overshoot":10, "landfill_under":150},
                          lead_times=None):
    regions = ["National"]
    data_minY, data_maxY = int(df_main["year"].min()), int(df_main["year"].max())

    def last_valid_before_or_equal(y, r):
        for yy in range(min(y, data_maxY), data_minY-1, -1):
            sub = df_main[(df_main["year"]==yy)&(df_main["region"]==r)]
            if not sub.empty:
                for col in ["waste_collected_ton_per_year","waste_generated_ton_per_year"]:
                    if col in sub.columns and sub[col].notna().any():
                        vv = safe_float(sub[col].iloc[0], np.nan)
                        if np.isfinite(vv): return vv
        return np.nan
    def first_valid_at_or_after(y, r):
        for yy in range(max(y, data_minY), data_maxY+1):
            sub = df_main[(df_main["year"]==yy)&(df_main["region"]==r)]
            if not sub.empty:
                for col in ["waste_collected_ton_per_year","waste_generated_ton_per_year"]:
                    if col in sub.columns and sub[col].notna().any():
                        vv = safe_float(sub[col].iloc[0], np.nan)
                        if np.isfinite(vv): return vv
        return np.nan

    W_expected={}
    for r in regions:
        ref_last = last_valid_before_or_equal(data_maxY, r)
        if not np.isfinite(ref_last): ref_last = 0.0
        for y in YEARS:
            if y < data_minY:
                base = first_valid_at_or_after(data_minY, r); base = 0.0 if not np.isfinite(base) else base
            elif y <= data_maxY:
                base = last_valid_before_or_equal(y, r)
                if not np.isfinite(base): base = first_valid_at_or_after(y, r)
                base = 0.0 if not np.isfinite(base) else base
                base *= sp["growth_multiplier"]
            else:
                delta = y - data_maxY
                base = ref_last * (sp["growth_multiplier"] ** delta)
            W_expected[(r,y)] = base
    W_use = {(r,y): safe_float(W_expected[(r,y)]*(1.0+sp.get("uncertainty_pct",0.0)),0.0)
             for r in regions for y in YEARS}

    def base_caps_func(y):
        if base_caps_by_year and y in base_caps_by_year:
            Rcap = float(base_caps_by_year[y].get("recycle",0.0) or 0.0)
            Tcap = float(base_caps_by_year[y].get("treatment",0.0) or 0.0)
        else:
            Wa = np.mean([W_use[(regions[0],yy)] for yy in YEARS]) if YEARS else 0.0
            Rcap = Wa*(1-landfill_share_for_base)*0.7
            Tcap = Wa*(1-landfill_share_for_base)*0.3
        Wa_y = W_use[(regions[0], y)]
        Lcap = max(Wa_y*landfill_share_for_base*1.05, 0.0)
        return Lcap, Rcap, Tcap

    prob = pl.LpProblem(f"DSS_dyn_v3_{scenario_name}", pl.LpMinimize)
    cat = pl.LpContinuous
    yL,yR,yT,nL,nR,nT,sW,sRdef,sTdef,sLexc,sRover,sTover,sLunder = {},{},{},{},{},{},{},{},{},{},{},{},{}
    for r in regions:
        for y in YEARS:
            yL[(r,y)] = pl.LpVariable(f"yL_{r}_{y}", lowBound=0)
            yR[(r,y)] = pl.LpVariable(f"yR_{r}_{y}", lowBound=0)
            yT[(r,y)] = pl.LpVariable(f"yT_{r}_{y}", lowBound=0)
            nL[(r,y)] = pl.LpVariable(f"nL_{r}_{y}", lowBound=0, cat=cat)
            nR[(r,y)] = pl.LpVariable(f"nR_{r}_{y}", lowBound=0, cat=cat)
            nT[(r,y)] = pl.LpVariable(f"nT_{r}_{y}", lowBound=0, cat=cat)
            sW[(r,y)]    = pl.LpVariable(f"sW_{r}_{y}",    lowBound=0)
            sRdef[(r,y)] = pl.LpVariable(f"sRdef_{r}_{y}", lowBound=0)
            sTdef[(r,y)] = pl.LpVariable(f"sTdef_{r}_{y}", lowBound=0)
            sLexc[(r,y)] = pl.LpVariable(f"sLexc_{r}_{y}", lowBound=0)
            sRover[(r,y)] = pl.LpVariable(f"sRover_{r}_{y}", lowBound=0)
            sTover[(r,y)] = pl.LpVariable(f"sTover_{r}_{y}", lowBound=0)
            sLunder[(r,y)] = pl.LpVariable(f"sLunder_{r}_{y}", lowBound=0)

    default_lead = {"landfill":0,"recycle":1,"treatment":1}
    if lead_times is None: lead_times = {}
    lead = {**default_lead, **lead_times}

    per = {"landfill":1_000_000,
           "recycle":  float((per_fac_override or {}).get("recycle", 120_000)),
           "treatment":float((per_fac_override or {}).get("treatment", 350_000))}

    def avail_cap(r,y, kind):
        baseL,baseR,baseT = base_caps_func(y)
        base_val = {"landfill":baseL,"recycle":baseR,"treatment":baseT}[kind]
        adds=[]
        for yy in YEARS:
            if yy + lead[kind] <= y:
                adds.append({"landfill":nL, "recycle":nR, "treatment":nT}[kind][(r,yy)] * per[kind])
        return base_val + (pl.lpSum(adds) if adds else 0)

    for r in regions:
        for y in YEARS:
            W = float(W_use[(r,y)])
            prob += yL[(r,y)] + yR[(r,y)] + yT[(r,y)] + sW[(r,y)] == W

            rt = (target_by_year or {}).get(y, None)
            rt = float(rt) if rt is not None and np.isfinite(rt) else sp["recycle_target"]
            prob += yR[(r,y)] + sRdef[(r,y)] >= rt * W
            prob += yR[(r,y)] - sRover[(r,y)] <= (rt + sp.get("target_band_up", 0.02)) * W

            if sp.get("min_treat_share", None) is not None:
                prob += yT[(r,y)] + sTdef[(r,y)] >= sp["min_treat_share"] * W
            prob += yL[(r,y)] - sLexc[(r,y)] <= sp["max_landfill_share"] * W

            if share_caps.get("recycle_max") is not None:
                prob += yR[(r,y)] - sRover[(r,y)] <= share_caps["recycle_max"] * W
            if share_caps.get("treat_max") is not None:
                prob += yT[(r,y)] - sTover[(r,y)] <= share_caps["treat_max"] * W
            if share_caps.get("landfill_min") is not None:
                prob += yL[(r,y)] + sLunder[(r,y)] >= share_caps["landfill_min"] * W

            prob += yL[(r,y)] <= avail_cap(r,y,"landfill")
            prob += yR[(r,y)] <= avail_cap(r,y,"recycle")
            prob += yT[(r,y)] <= avail_cap(r,y,"treatment")

            if ramp_limits:
                prob += nR[(r,y)] <= max_new_per_year.get("recycle", 9999)
                prob += nT[(r,y)] <= max_new_per_year.get("treatment", 9999)
                prob += nL[(r,y)] <= max_new_per_year.get("landfill", 9999)

    DISCOUNT_RATE = sp.get("discount_rate", 0.08)
    ANNUALIZED_CAPEX_MUSD = {"landfill":5.0,"recycle":3.0,"treatment":12.0}
    OPEX_USD_PER_TON     = {"landfill":30.0,"recycle":32.0,"treatment":40.0}
    EMISSION_FACTORS     = {"landfill":480.0,"recycle":60.0,"treatment":200.0}
    SLACK_PENALTIES      = sp.get("slack_penalties", {"unserved_waste":1000,"recycle_deficit":500,"treat_deficit":1200,"landfill_excess":400})

    obj=[]
    for r in regions:
        for y in YEARS:
            dfac = 1/((1+DISCOUNT_RATE)**(y - YEARS[0]))
            capex = (nL[(r,y)]*ANNUALIZED_CAPEX_MUSD["landfill"] +
                     nR[(r,y)]*ANNUALIZED_CAPEX_MUSD["recycle"]  +
                     nT[(r,y)]*ANNUALIZED_CAPEX_MUSD["treatment"])
            opex  = (yL[(r,y)]*OPEX_USD_PER_TON["landfill"] +
                     yR[(r,y)]*OPEX_USD_PER_TON["recycle"]  +
                     yT[(r,y)]*OPEX_USD_PER_TON["treatment"]) / 1_000_000.0
            co2kg = (yL[(r,y)]*EMISSION_FACTORS["landfill"] +
                     yR[(r,y)]*EMISSION_FACTORS["recycle"]  +
                     yT[(r,y)]*EMISSION_FACTORS["treatment"])
            carbon = ((co2kg/1000.0) * sp.get("carbon_price",0.0)) / 1_000_000.0

            pen_base = (sW[(r,y)]*SLACK_PENALTIES["unserved_waste"] +
                        sRdef[(r,y)]*SLACK_PENALTIES["recycle_deficit"] +
                        sTdef[(r,y)]*SLACK_PENALTIES["treat_deficit"] +
                        sLexc[(r,y)]*SLACK_PENALTIES["landfill_excess"]) / 1_000_000.0

            pen_extra = (sRover[(r,y)]*extra_penalties.get("recycle_overshoot",0) +
                         sTover[(r,y)]*extra_penalties.get("treat_overshoot",0) +
                         sLunder[(r,y)]*extra_penalties.get("landfill_under",0)) / 1_000_000.0

            obj.append(dfac*(capex+opex+carbon+pen_base+pen_extra))
    prob += pl.lpSum(obj)
    _ = prob.solve(pl.PULP_CBC_CMD(msg=False))

    rowsF,rowsN,rowsC,rowsS = [],[],[],[]
    for r in regions:
        for y in YEARS:
            baseL,baseR,baseT = base_caps_func(y)
            rowsF.append({"scenario":scenario_name,"region":r,"year":y,
                          "W_used_ton":W_use[(r,y)],
                          "y_landfill":pl.value(yL[(r,y)]),
                          "y_recycle": pl.value(yR[(r,y)]),
                          "y_treat":   pl.value(yT[(r,y)])})
            rowsN.append({"scenario":scenario_name,"region":r,"year":y,
                          "new_landfills":pl.value(nL[(r,y)]),
                          "new_recycle_plants":pl.value(nR[(r,y)]),
                          "new_treatment_plants":pl.value(nT[(r,y)] )})
            rowsC.append({"scenario":scenario_name,"region":r,"year":y,
                          "base_cap_landfill":baseL,"base_cap_recycle":baseR,"base_cap_treatment":baseT})
            rowsS.append({"scenario":scenario_name,"region":r,"year":y,
                          "s_unserved":pl.value(sW[(r,y)]),
                          "s_recycle_def":pl.value(sRdef[(r,y)]),
                          "s_treat_def":pl.value(sTdef[(r,y)]),
                          "s_landfill_exc":pl.value(sLexc[(r,y)]),
                          "s_recycle_overshoot":pl.value(sRover[(r,y)]),
                          "s_treat_overshoot":pl.value(sTover[(r,y)]),
                          "s_landfill_under":pl.value(sLunder[(r,y)])})
    total_cost = float(pl.value(pl.lpSum(obj)))
    return (pd.DataFrame(rowsF), pd.DataFrame(rowsN], pd.DataFrame(rowsC),
            total_cost, pd.DataFrame(rowsS))

# =========================
# UI (TEK SIDEBAR BLOĞU)
# =========================
st.title("♻️ Türkiye Belediye Atıkları — RF + DSS v3 (Streamlit)")

with st.sidebar:
    st.header("1) Veriler")
    use_repo = st.toggle("Repo verisini kullan (önerilir)", value=True,
                         help="data/ klasöründeki dosyaları otomatik yükler.")
    up_main = up_fac = None
    if not use_repo:
        up_main = st.file_uploader("Ulusal veri (xlsx/csv)", type=["xlsx","xls","csv"], key="main")
        up_fac  = st.file_uploader("Tesis tablosu (xlsx/csv) — RF impute", type=["xlsx","xls","csv"], key="fac")

    # Ulusal veri
    if use_repo and DEFAULT_MAIN_PATH.exists():
        df_main = load_main_table(DEFAULT_MAIN_PATH)
        st.caption(f"Ulusal veri: `{DEFAULT_MAIN_PATH.name}` (repo)")
    elif up_main is not None:
        df_main = load_main_table(up_main)
        st.caption(f"Ulusal veri: yüklenen dosya (`{up_main.name}`)")
    else:
        df_main = SAMPLE_DF.copy()
        st.caption("Ulusal veri: örnek dataset kullanılıyor")

    if df_main is None or df_main.empty or ("year" not in df_main.columns):
        st.warning("Ulusal veri geçersiz — örnek veri yüklendi.")
        df_main = SAMPLE_DF.copy()

    # Tesis tablosu
    if use_repo and DEFAULT_FAC_PATH.exists():
        dfF = load_facility_table(DEFAULT_FAC_PATH)
        st.caption(f"Tesis tablosu: `{DEFAULT_FAC_PATH.name}` (repo)")
    elif up_fac is not None:
        dfF = load_facility_table(up_fac)
        st.caption(f"Tesis tablosu: yüklenen dosya (`{up_fac.name}`)")
    else:
        dfF = None
        st.caption("Tesis tablosu: (opsiyonel) — yoksa UI’daki kapasite ayarları kullanılır")

    # === Zaman ufku ===
    st.header("2) Zaman Ufku")

    year_vals = pd.to_numeric(df_main["year"], errors="coerce").dropna()
    year_vals = year_vals[year_vals.between(1900, 2100)]
    if year_vals.empty:
        data_minY, data_maxY = DEFAULT_START_YEAR, DEFAULT_END_YEAR
        st.warning("Geçerli yıl bulunamadı — 2020–2025 varsayılan aralık kullanılacak.")
    else:
        data_minY, data_maxY = int(year_vals.min()), int(year_vals.max())

    if "start_year" not in st.session_state:
        st.session_state["start_year"] = DEFAULT_START_YEAR
    if "end_year" not in st.session_state:
        st.session_state["end_year"] = DEFAULT_END_YEAR

    start_y = st.number_input("Başlangıç yılı",
                              value=int(st.session_state["start_year"]),
                              step=1, key="start_year")
    end_y   = st.number_input("Bitiş yılı",
                              value=int(st.session_state["end_year"]),
                              step=1, key="end_year")

    if end_y <= start_y:
        end_y = int(start_y) + 1

    YEARS = list(range(int(start_y), int(end_y)+1))
    st.caption(f"Verideki aralık: {data_minY}–{data_maxY} | Seçim: {int(start_y)}–{int(end_y)}")

    # === PRESET BLOĞU ===
    st.header("0) Senaryo Presetleri")
    colp1, colp2 = st.columns([3,1])
    with colp1:
        preset_choice = st.selectbox("Preset seç", ["Custom","Baseline","Budget","Green"], index=0)
    with colp2:
        if st.button("Uygula"):
            if preset_choice != "Custom":
                apply_preset_to_session(preset_choice)
                st.rerun()

    # İlk açılış varsayılanları
    _set_if_missing("sp_recycle_target", 0.60)
    _set_if_missing("sp_min_treat",     0.15)
    _set_if_missing("sp_max_landfill",  0.60)
    _set_if_missing("sp_target_band",   0.02)
    _set_if_missing("sp_carbon_price",  100.0)
    _set_if_missing("sp_growth",        1.00)
    _set_if_missing("sp_uncertainty",   0.10)
    _set_if_missing("sp_discount",      0.08)

    _set_if_missing("cap_recycle_max",  0.85)
    _set_if_missing("cap_treat_max",    0.35)
    _set_if_missing("cap_landfill_min", 0.02)
    _set_if_missing("pen_overshoot",    600.0)
    _set_if_missing("pen_landfill_under", 150.0)

    _set_if_missing("per_recycle", 120_000.0)
    _set_if_missing("per_treat",   350_000.0)
    _set_if_missing("max_new_R",   10)
    _set_if_missing("max_new_T",   22)
    _set_if_missing("lead_recycle", 0)
    _set_if_missing("lead_treatment", 1)

    st.header("3) Politika Hedefleri")
    sp = dict(
        recycle_target   = st.slider("Sabit geri dönüşüm hedefi (yoksa EU serisi kullanılır)",
                                     0.0, 0.95, value=st.session_state["sp_recycle_target"], step=0.01, key="sp_recycle_target"),
        min_treat_share  = st.slider("Arıtma min payı",
                                     0.00, 0.50, value=st.session_state["sp_min_treat"], step=0.01, key="sp_min_treat"),
        max_landfill_share = st.slider("Depolama max payı",
                                     0.00, 1.00, value=st.session_state["sp_max_landfill"], step=0.01, key="sp_max_landfill"),
        target_band_up   = st.slider("Hedef üst bandı (+ puan)",
                                     0.00, 0.05, value=st.session_state["sp_target_band"], step=0.01, key="sp_target_band"),
        carbon_price     = st.number_input("Karbon fiyatı ($/tCO2e)",
                                     value=float(st.session_state["sp_carbon_price"]), step=10.0, key="sp_carbon_price"),
        growth_multiplier= st.slider("Yıllık büyüme katsayısı",
                                     0.90, 1.10, value=float(st.session_state["sp_growth"]), step=0.01, key="sp_growth"),
        uncertainty_pct  = st.slider("Belirsizlik (+%)",
                                     0.0, 0.30, value=float(st.session_state["sp_uncertainty"]), step=0.01, key="sp_uncertainty"),
        discount_rate    = st.slider("İskonto oranı",
                                     0.00, 0.20, value=float(st.session_state["sp_discount"]), step=0.01, key="sp_discount"),
        slack_penalties  = {"unserved_waste": 1000, "recycle_deficit": 500, "treat_deficit": 1200, "landfill_excess": 400}
    )

    st.header("4) Pay Sınırları ve Cezalar")
    recycle_max = st.slider("Geri dönüşüm üst tavanı", 0.50, 0.95, value=st.session_state["cap_recycle_max"], step=0.01, key="cap_recycle_max")
    treat_max   = st.slider("Arıtma üst tavanı",       0.20, 0.60, value=st.session_state["cap_treat_max"],   step=0.01, key="cap_treat_max")
    landfill_min= st.slider("Depolama alt tabanı",     0.00, 0.10, value=st.session_state["cap_landfill_min"],step=0.01, key="cap_landfill_min")
    overshoot_pen = st.number_input("Overshoot cezası ($/ton) [geri dönüşüm]", value=float(st.session_state["pen_overshoot"]), step=50.0, key="pen_overshoot")
    landfill_under_pen = st.number_input("Depolama altı cezası ($/ton)", value=float(st.session_state["pen_landfill_under"]), step=10.0, key="pen_landfill_under")

    st.header("5) Kapasite ve Kurulum")
    per_recycle = st.number_input("MRF kapasitesi (t/y)",  value=float(st.session_state["per_recycle"]), step=10_000.0, min_value=10_000.0, key="per_recycle")
    per_treat   = st.number_input("Arıtma kapasitesi (t/y)", value=float(st.session_state["per_treat"]),   step=10_000.0, min_value=50_000.0, key="per_treat")

    max_new_R = st.number_input("Yıllık max yeni MRF (adet)", value=int(st.session_state["max_new_R"]), step=1, min_value=0, key="max_new_R")
    max_new_T = st.number_input("Yıllık max yeni arıtma (adet)", value=int(st.session_state["max_new_T"]), step=1, min_value=0, key="max_new_T")
    lead_recycle   = st.selectbox("MRF lead-time (yıl)", [0,1], index=[0,1].index(int(st.session_state["lead_recycle"])),     key="lead_recycle")
    lead_treatment = st.selectbox("Arıtma lead-time (yıl)", [0,1], index=[0,1].index(int(st.session_state["lead_treatment"])), key="lead_treatment")

    run_btn = st.button("▶︎ Modeli Çalıştır")

    cfg = {
        "preset": preset_choice,
        "years": [int(start_y), int(end_y)],
        "policy": {
            "recycle_target": st.session_state["sp_recycle_target"],
            "min_treat_share": st.session_state["sp_min_treat"],
            "max_landfill_share": st.session_state["sp_max_landfill"],
            "target_band_up": st.session_state["sp_target_band"],
            "carbon_price": st.session_state["sp_carbon_price"],
            "growth_multiplier": st.session_state["sp_growth"],
            "uncertainty_pct": st.session_state["sp_uncertainty"],
            "discount_rate": st.session_state["sp_discount"],
        },
        "share_caps": {
            "recycle_max": st.session_state["cap_recycle_max"],
            "treat_max": st.session_state["cap_treat_max"],
            "landfill_min": st.session_state["cap_landfill_min"],
        },
        "penalties_extra": {
            "recycle_overshoot": st.session_state["pen_overshoot"],
            "landfill_under": st.session_state["pen_landfill_under"],
            "treat_overshoot": 10.0,
        },
        "per_fac": {"recycle": st.session_state["per_recycle"], "treatment": st.session_state["per_treat"]},
        "ramp_limits": {"max_new_R": int(st.session_state["max_new_R"]), "max_new_T": int(st.session_state["max_new_T"])},
        "lead_times": {"recycle": int(st.session_state["lead_recycle"]), "treatment": int(st.session_state["lead_treatment"]), "landfill": 0},
        "use_repo": use_repo,
    }
    st.download_button("⬇️ Parametreleri JSON indir", data=json.dumps(cfg, indent=2),
                       file_name=f"config_{preset_choice.lower()}.json", mime="application/json")

st.subheader("Ulusal veri (ilk 10 satır)")
st.dataframe(df_main.head(10), use_container_width=True)

# ========= RUN =========
if run_btn:
    with st.spinner("RF imputation + kalibrasyon + DSS çözülüyor..."):
        target_by_year = build_target_series_from_facility(dfF) if dfF is not None else None
        if dfF is not None:
            base_caps_by_year, per_fac_from_data = capacities_by_year_from_facility(dfF)
        else:
            base_caps_by_year, per_fac_from_data = (None, {"recycle": per_recycle, "treatment": per_treat})

        per_fac_override = {
            "recycle":   float(per_recycle) if np.isfinite(per_recycle)
                         else float(per_fac_from_data.get("recycle", PER_RECYCLE_DEFAULT)),
            "treatment": float(per_treat)   if np.isfinite(per_treat)
                         else float(per_fac_from_data.get("treatment", PER_TREAT_DEFAULT)),
        }

        share_caps = {"recycle_max": float(recycle_max), "treat_max": float(treat_max), "landfill_min": float(landfill_min)}
        extra_pen  = {"recycle_overshoot": float(overshoot_pen), "treat_overshoot": 10.0, "landfill_under": float(landfill_under_pen)}
        lead_times = {"recycle": int(lead_recycle), "treatment": int(lead_treatment), "landfill": 0}

        flows_df, new_df, caps_df, npv_cost, slack_df = solve_with_dynamic_v3(
            df_main, "Green_v3_streamlit", sp, YEARS,
            target_by_year=target_by_year,
            per_fac_override=per_fac_override,
            base_caps_by_year=base_caps_by_year,
            ramp_limits=True,
            max_new_per_year={"recycle": int(max_new_R), "treatment": int(max_new_T), "landfill": 0},
            share_caps=share_caps,
            target_band_up=sp["target_band_up"],
            extra_penalties=extra_pen,
            lead_times=lead_times
        )

    st.success(f"NPV toplam maliyet: **{npv_cost:,.2f} M$**")

    g = flows_df.groupby("year").agg(
        yL=("y_landfill","sum"), yR=("y_recycle","sum"), yT=("y_treat","sum"), W=("W_used_ton","sum")
    ).reset_index()
    g["Landfill"] = g["yL"]/g["W"]; g["Recycle"] = g["yR"]/g["W"]; g["Treatment"] = g["yT"]/g["W"]
    area_df = g.melt(id_vars="year", value_vars=["Landfill","Recycle","Treatment"], var_name="Flow", value_name="Share")
    fig1 = px.area(area_df, x="year", y="Share", color="Flow", title="National Flow Shares — DSS v3")
    fig1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

    adds = new_df.groupby("year")[["new_landfills","new_recycle_plants","new_treatment_plants"]].sum().reset_index()
    adds = adds.rename(columns={"new_landfills":"Landfill","new_recycle_plants":"Recycle","new_treatment_plants":"Treatment"})
    adds_m = adds.melt(id_vars="year", var_name="Facility", value_name="Count")
    fig2 = px.bar(adds_m, x="year", y="Count", color="Facility", barmode="group", title="New Facilities per Year")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Slack toplamları (ton)")
    st.dataframe(
        slack_df.groupby("year")[["s_unserved","s_recycle_def","s_treat_def","s_landfill_exc",
                                  "s_recycle_overshoot","s_treat_overshoot","s_landfill_under"]].sum(),
        use_container_width=True
    )

    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Akışlar CSV", flows_df.to_csv(index=False).encode("utf-8"),
                           file_name="flows_v3.csv", mime="text/csv")
    with c2:
        st.download_button("⬇️ Yeni Tesisler CSV", new_df.to_csv(index=False).encode("utf-8"),
                           file_name="new_facilities_v3.csv", mime="text/csv")
    with c3:
        st.download_button("⬇️ Slack CSV", slack_df.to_csv(index=False).encode("utf-8"),
                           file_name="slack_v3.csv", mime="text/csv")
else:
    st.info("Soldan verileri/parametreleri seçin ve **Modeli Çalıştır**’a basın. Repo modu açıksa data/ klasöründeki dosyalar otomatik yüklenir.")

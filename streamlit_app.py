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

st.set_page_config(page_title="MSW DSS — Türkiye (RF + DSS v3)", page_icon="♻️", layout="wide")

# =========================
# Utilities & Defaults
# =========================
def _clean_decimal_series(s: pd.Series) -> pd.Series:
    # "30,13" -> 30.13; boş->NaN
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
    "year": [r"\byear\b", r"\byil\b", r"\byıl\b"],
    "region": [r"\bregion\b", r"\bbolge\b", r"\bbolge\b"],
    "waste_collected_ton_per_year": [
        r"waste.*collec", r"toplanan.*atik", r"toplanan_atik", r"collected.*ton"
    ],
    "waste_generated_ton_per_year": [
        r"waste.*generat", r"uretilen.*atik", r"üretilen.*atik", r"generated.*ton"
    ],
    # Kapasite kolonları opsiyonel; varsa okunur
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
    # Dosyayı oku
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

    # Yıl kolonu bul (year / yil / yıl / ilk sütun fallback)
    year_col = _find_first_match(raw.columns, MAIN_SYNONYMS["year"])
    if year_col is None:
        # ilk sütun numarik ve 1900-2100 aralığında ise yıl varsay
        first = raw.columns[0]
        cand = pd.to_numeric(raw[first], errors="coerce")
        if cand.between(1900, 2100).any():
            year_col = first
    if year_col is None:
        st.error("Ulusal veri: Yıl kolonu (year/yıl/yil) bulunamadı.")
        st.stop()

    # Yılı güvenli üret (yy/mm/dd, "2019*", vb varsa 4 haneliyi çekiyoruz)
    y = pd.to_numeric(
            raw[year_col].astype(str).str.extract(r"(\d{4})")[0],
            errors="coerce"
        )
    
    df = raw.copy()
    # Eski 'year' kolonunu tamamen kaldırıp temiz seriyi ekleyelim
    if year_col in df.columns and year_col != "year":
        df = df.rename(columns={year_col: "year"})
    if "year" in df.columns:
        df = df.drop(columns=["year"])
    df["year"] = y
    
    # NaN yıl satırlarını at ve int'e çevir
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].round().astype(int)



    # Region yoksa National ekle
    reg_col = _find_first_match(df.columns, MAIN_SYNONYMS["region"])
    if reg_col and reg_col != "region":
        df.rename(columns={reg_col: "region"}, inplace=True)
    if "region" not in df.columns:
        df["region"] = "National"

    # İçerik kolonlarını eşleştir
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
            out[std_key] = _clean_decimal_series(df[src])

    # En az bir talep kolonu zorunlu
    if not (("waste_collected_ton_per_year" in out.columns) or
            ("waste_generated_ton_per_year" in out.columns)):
        st.error("Ulusal veri: Toplanan/Üretilen atık kolonlarından en az biri bulunmalı.")
        st.stop()

    return out.sort_values(["year","region"]).reset_index(drop=True)

# -------------------------------------------------------------------
# B) TESIS TABLOSU (RF IMPUTE)  — aynı kaldı
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

    # numerik dönüşüm
    for c in dfF.columns:
        if c == "year": continue
        dfF[c] = _clean_decimal_series(dfF[c])

    # lag özellikleri
    dfF = dfF.sort_values("year")
    num_cols = [c for c in dfF.columns if c!="year"]
    for c in num_cols:
        dfF[f"{c}_lag1"] = dfF[c].shift(1)

    # RF imputation
    rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    imputer = IterativeImputer(estimator=rf, random_state=42, max_iter=20)
    imputed = dfF.copy()
    imp_cols = num_cols + [c for c in dfF.columns if c.endswith("_lag1")]
    imputed[imp_cols] = imputer.fit_transform(imputed[imp_cols])

    # tesis sayıları tamsayı ve >=0
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
# D) DSS v3 (aynı – paylaşım kısıtları, ramp, lead-times, vb.)
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
    return (pd.DataFrame(rowsF), pd.DataFrame(rowsN), pd.DataFrame(rowsC),
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
    y = pd.to_numeric(df_main["year"], errors="coerce")
    # Mevcut int serinin üstüne NaN yazmamak için drop + ekle:
    df_main = df_main.drop(columns=["year"])
    df_main["year"] = y
    df_main = df_main[df_main["year"].notna()].copy()
    df_main["year"] = df_main["year"].round().astype(int)

    minY = int(df_main["year"].min()); maxY = max(int(df_main["year"].max()), minY+5)
    start_y = st.number_input("Başlangıç yılı", value=minY, step=1)
    end_y   = st.number_input("Bitiş yılı", value=maxY, step=1)
    YEARS = list(range(int(start_y), int(end_y)+1))

    # === Politika/Hedefler ===
    st.header("3) Politika Hedefleri")
    sp = dict(
        recycle_target = st.slider("Sabit geri dönüşüm hedefi (yoksa EU serisi kullanılır)", 0.0, 0.95, 0.60, 0.01),
        min_treat_share= st.slider("Arıtma min payı", 0.00, 0.50, 0.15, 0.01),
        max_landfill_share = st.slider("Depolama max payı", 0.00, 1.00, 0.60, 0.01),
        target_band_up = st.slider("Hedef üst bandı (+ puan)", 0.00, 0.05, 0.02, 0.01),
        carbon_price = st.number_input("Karbon fiyatı ($/tCO2e)", value=100.0, step=10.0),
        growth_multiplier = st.slider("Yıllık büyüme katsayısı", 0.90, 1.10, 1.00, 0.01),
        uncertainty_pct   = st.slider("Belirsizlik (+%)", 0.0, 0.30, 0.10, 0.01),
        discount_rate     = st.slider("İskonto oranı", 0.00, 0.20, 0.08, 0.01),
        slack_penalties   = {"unserved_waste": 1000, "recycle_deficit": 500, "treat_deficit": 1200, "landfill_excess": 400}
    )

    st.header("4) Pay Sınırları ve Cezalar")
    recycle_max = st.slider("Geri dönüşüm üst tavanı", 0.50, 0.95, 0.85, 0.01)
    treat_max   = st.slider("Arıtma üst tavanı", 0.20, 0.60, 0.35, 0.01)
    landfill_min= st.slider("Depolama alt tabanı", 0.00, 0.10, 0.02, 0.01)
    overshoot_pen = st.number_input("Overshoot cezası ($/ton) [geri dönüşüm]", value=600.0, step=50.0)
    landfill_under_pen = st.number_input("Depolama altı cezası ($/ton)", value=150.0, step=10.0)

    st.header("5) Kapasite ve Kurulum")
    per_recycle_default = 120_000.0
    per_treat_default   = 350_000.0
    per_recycle = st.number_input("MRF kapasitesi (t/y)", value=per_recycle_default, step=10_000.0, min_value=10_000.0)
    per_treat   = st.number_input("Arıtma kapasitesi (t/y)", value=per_treat_default, step=10_000.0, min_value=50_000.0)

    max_new_R = st.number_input("Yıllık max yeni MRF (adet)", value=10, step=1, min_value=0)
    max_new_T = st.number_input("Yıllık max yeni arıtma (adet)", value=22, step=1, min_value=0)
    lead_recycle   = st.selectbox("MRF lead-time (yıl)", [0,1], index=0)
    lead_treatment = st.selectbox("Arıtma lead-time (yıl)", [0,1], index=0)

    run_btn = st.button("▶︎ Modeli Çalıştır")

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
            "recycle":   float(per_recycle if per_recycle else per_fac_from_data.get("recycle", per_recycle_default)),
            "treatment": float(per_treat   if per_treat   else per_fac_from_data.get("treatment", per_treat_default)),
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
    area_df = g.melt(id_vars="year", value_vars=["Landfill","Recycle","Treatment"],
                     var_name="Flow", value_name="Share")
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

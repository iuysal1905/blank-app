import io, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pulp as pl

st.set_page_config(page_title="MSW DSS — Türkiye", page_icon="♻️", layout="wide")

# -------------------- Helper / Defaults --------------------
DEFAULT_PRESETS = {
    "Baseline": dict(
        recycle_target=0.45, min_treat_share=0.10, max_landfill_share=0.60,
        growth_multiplier=1.00, uncertainty_pct=0.00, carbon_price=10.0
    ),
    "Green": dict(
        recycle_target=0.85, min_treat_share=0.15, max_landfill_share=0.05,
        growth_multiplier=1.00, uncertainty_pct=0.10, carbon_price=100.0
    ),
    "Budget": dict(
        recycle_target=0.30, min_treat_share=0.10, max_landfill_share=0.60,
        growth_multiplier=1.00, uncertainty_pct=0.05, carbon_price=0.0
    )
}

# Kapasiteler (ton/yıl) — örnek kalibrasyon (geri dönüşüm ~80k/tesis, arıtma ~320k/tesis)
PER_FACILITY_CAP = {"landfill": 1_000_000, "recycle": 80_000, "treatment": 320_000}
FACILITY_LEAD_TIME_YEARS = {"landfill": 0, "recycle": 1, "treatment": 1}

# Maliyetler
ANNUALIZED_CAPEX_MUSD = {"landfill": 5.0, "recycle": 3.0, "treatment": 12.0}  # M$ / tesis-yıl
OPEX_USD_PER_TON     = {"landfill": 30.0, "recycle": 32.0, "treatment": 40.0} # $/ton
DISCOUNT_RATE = 0.08

# Emisyon faktörleri (kgCO2e/ton) — isterseniz yan panelden değiştirebilirsiniz
EMISSION_FACTORS = {"landfill": 480.0, "recycle": 60.0, "treatment": 200.0}

# Başlangıç payları ve başlık boşluğu (veride kapasite yoksa kullanılır)
BASELINE_SHARES = {"landfill": 0.65, "recycle": 0.25, "treatment": 0.10}
BASELINE_HEADROOM = 1.05

# Slack cezaları ($/ton) — yatırım yerine ceza ödemesini caydırır
SLACK_PENALTIES = {
    "unserved_waste": 1000,
    "recycle_deficit": 300,
    "treat_deficit": 300,
    "landfill_excess": 100,
}

USE_INTEGER_FACILITIES = False     # integer isterseniz True yapın
CAPACITY_ANCHOR = "last_data"      # ya da bir yıl (int)

# -------------------- Data I/O --------------------
ST_SAMPLE_DF = pd.DataFrame({
    "region": ["National", "National", "National"],
    "year":   [2020, 2021, 2022],
    # Aşağıdaki iki kolondan biri yeterli:
    "waste_collected_ton_per_year": [30_000_000, np.nan, 30_283_760],
    "waste_generated_ton_per_year": [32_000_000, 33_940_700, 31_797_940],
    # Kapasiteler varsa daha doğru başlar:
    "landfill_capacity_ton":  [23_848_460, np.nan, np.nan],
    "recycle_capacity_ton":   [4_769_692,  np.nan, np.nan],
    "treatment_capacity_ton": [3_179_794,  np.nan, np.nan],
})

def read_input_df(uploaded):
    if uploaded is None:
        return ST_SAMPLE_DF.copy()
    ext = uploaded.name.lower().split(".")[-1]
    if ext in ("xlsx","xls"):
        df = pd.read_excel(uploaded)
    elif ext in ("csv",):
        df = pd.read_csv(uploaded)
    else:
        st.error("Desteklenen formatlar: .xlsx, .xls, .csv")
        st.stop()
    return df

# -------------------- Core Solver --------------------
def solve_multiyear_scenario(df, scenario_name, sp, years):
    """Slack'li tek amaç: NPV maliyet + karbon maliyeti. 5 output döner."""
    VERSION_TAG = "v2-slack-5ret"
    regions = ["National"]
    data_minY, data_maxY = int(df["year"].min()), int(df["year"].max())

    def safe(x, default=0.0):
        try:
            v=float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def last_valid_before_or_equal(y, r):
        for yy in range(min(y, data_maxY), data_minY-1, -1):
            sub = df[(df["year"]==yy)&(df["region"]==r)]
            if not sub.empty:
                for col in ["waste_collected_ton_per_year","waste_generated_ton_per_year"]:
                    if col in sub.columns and sub[col].notna().any():
                        v=safe(sub[col].iloc[0], np.nan)
                        if np.isfinite(v): return v
        return np.nan

    def first_valid_at_or_after(y, r):
        for yy in range(max(y, data_minY), data_maxY+1):
            sub = df[(df["year"]==yy)&(df["region"]==r)]
            if not sub.empty:
                for col in ["waste_collected_ton_per_year","waste_generated_ton_per_year"]:
                    if col in sub.columns and sub[col].notna().any():
                        v=safe(sub[col].iloc[0], np.nan)
                        if np.isfinite(v): return v
        return np.nan

    # Talep
    W_expected = {}
    for r in regions:
        ref_last = last_valid_before_or_equal(data_maxY, r)
        if not np.isfinite(ref_last): ref_last = 0.0
        for y in years:
            if y < data_minY:
                base = first_valid_at_or_after(data_minY, r); base = 0.0 if not np.isfinite(base) else base
                W_expected[(r,y)] = base
            elif y <= data_maxY:
                base = last_valid_before_or_equal(y, r)
                if not np.isfinite(base): base = first_valid_at_or_after(y, r)
                base = 0.0 if not np.isfinite(base) else base
                W_expected[(r,y)] = base * sp["growth_multiplier"]
            else:
                delta = y - data_maxY
                W_expected[(r,y)] = ref_last * (sp["growth_multiplier"] ** delta)
    W_use = {(r,y): safe(W_expected[(r,y)]*(1.0+sp.get("uncertainty_pct",0.0)),0.0)
             for r in regions for y in years}

    # Kapasite ankoru
    if CAPACITY_ANCHOR == "last_data": anchor_year = data_maxY
    elif isinstance(CAPACITY_ANCHOR, int): anchor_year = int(CAPACITY_ANCHOR)
    else: anchor_year = data_maxY

    def anchor_caps_for_region(r):
        sub = df[(df["region"]==r) & (df["year"]==anchor_year)]
        if not sub.empty and {"landfill_capacity_ton","recycle_capacity_ton","treatment_capacity_ton"}.issubset(sub.columns):
            L = safe(sub["landfill_capacity_ton"].iloc[0], np.nan)
            R = safe(sub["recycle_capacity_ton"].iloc[0], np.nan)
            T = safe(sub["treatment_capacity_ton"].iloc[0], np.nan)
            if np.isfinite(L) and np.isfinite(R) and np.isfinite(T) and (L+R+T)>0:
                return L,R,T
        W_anchor = safe(W_expected[(r, min(max(anchor_year, years[0]), years[-1]))], 0.0)
        return (W_anchor*BASELINE_SHARES["landfill"]*BASELINE_HEADROOM,
                W_anchor*BASELINE_SHARES["recycle"] *BASELINE_HEADROOM,
                W_anchor*BASELINE_SHARES["treatment"]*BASELINE_HEADROOM)

    base_cap = {}
    for r in regions:
        caps_r = anchor_caps_for_region(r)
        for y in years:
            base_cap[(r,y)] = caps_r

    # Model
    prob = pl.LpProblem(f"DSS_{scenario_name}", pl.LpMinimize)
    cat = pl.LpInteger if USE_INTEGER_FACILITIES else pl.LpContinuous

    yL,yR,yT,nL,nR,nT = {},{},{},{},{},{}
    sW, sRdef, sTdef, sLexc = {},{},{},{}
    for r in regions:
        for y in years:
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

    def avail_cap(r,y, kind):
        baseL,baseR,baseT = base_cap[(r,y)]
        base_val = {"landfill":baseL,"recycle":baseR,"treatment":baseT}[kind]
        lead = FACILITY_LEAD_TIME_YEARS[kind]
        add_list=[]
        for yy in years:
            if yy + lead <= y:
                if kind=="landfill": add_list.append(nL[(r,yy)] * PER_FACILITY_CAP[kind])
                if kind=="recycle":  add_list.append(nR[(r,yy)] * PER_FACILITY_CAP[kind])
                if kind=="treatment":add_list.append(nT[(r,yy)] * PER_FACILITY_CAP[kind])
        return base_val + (pl.lpSum(add_list) if add_list else 0)

    for r in regions:
        for y in years:
            W = float(W_use[(r,y)])
            # Talep denklemi
            prob += yL[(r,y)] + yR[(r,y)] + yT[(r,y)] + sW[(r,y)] == W
            # Politika hedefleri
            prob += yR[(r,y)] + sRdef[(r,y)] >= sp["recycle_target"] * W
            if sp.get("min_treat_share", None) is not None:
                prob += yT[(r,y)] + sTdef[(r,y)] >= sp["min_treat_share"] * W
            prob += yL[(r,y)] - sLexc[(r,y)] <= sp["max_landfill_share"] * W
            # Kapasite
            prob += yL[(r,y)] <= avail_cap(r,y,"landfill")
            prob += yR[(r,y)] <= avail_cap(r,y,"recycle")
            prob += yT[(r,y)] <= avail_cap(r,y,"treatment")

    # Amaç (NPV maliyet + karbon)
    obj_terms=[]
    for r in regions:
        for y in years:
            dfac = 1.0/((1.0+DISCOUNT_RATE)**(y - years[0]))
            capex = (nL[(r,y)]*ANNUALIZED_CAPEX_MUSD["landfill"] +
                     nR[(r,y)]*ANNUALIZED_CAPEX_MUSD["recycle"]  +
                     nT[(r,y)]*ANNUALIZED_CAPEX_MUSD["treatment"])
            opex  = (yL[(r,y)]*OPEX_USD_PER_TON["landfill"] +
                     yR[(r,y)]*OPEX_USD_PER_TON["recycle"]  +
                     yT[(r,y)]*OPEX_USD_PER_TON["treatment"]) / 1_000_000.0
            # Emisyon ve karbon
            co2kg = (yL[(r,y)]*EMISSION_FACTORS["landfill"] +
                     yR[(r,y)]*EMISSION_FACTORS["recycle"]  +
                     yT[(r,y)]*EMISSION_FACTORS["treatment"])
            carbon = ((co2kg/1000.0) * sp.get("carbon_price",0.0)) / 1_000_000.0
            # Slack cezaları
            pen = (sW[(r,y)]*SLACK_PENALTIES["unserved_waste"] +
                   sRdef[(r,y)]*SLACK_PENALTIES["recycle_deficit"] +
                   sTdef[(r,y)]*SLACK_PENALTIES["treat_deficit"] +
                   sLexc[(r,y)]*SLACK_PENALTIES["landfill_excess"]) / 1_000_000.0
            obj_terms.append(dfac*(capex + opex + carbon + pen))

    prob += pl.lpSum(obj_terms)
    _ = prob.solve(pl.PULP_CBC_CMD(msg=False))

    # Çıktılar
    rowsF, rowsN, rowsC, rowsS = [], [], [], []
    for r in regions:
        for y in years:
            Lcap = pl.value(avail_cap(r,y,"landfill"))
            Rcap = pl.value(avail_cap(r,y,"recycle"))
            Tcap = pl.value(avail_cap(r,y,"treatment"))
            rowsF.append({"scenario": scenario_name, "region": r, "year": y,
                          "W_expected_ton": float(W_expected[(r,y)]),
                          "W_used_ton": float(W_use[(r,y)]),
                          "y_landfill": pl.value(yL[(r,y)]),
                          "y_recycle":  pl.value(yR[(r,y)]),
                          "y_treat":    pl.value(yT[(r,y)])})
            rowsN.append({"scenario": scenario_name, "region": r, "year": y,
                          "new_landfills": pl.value(nL[(r,y)]),
                          "new_recycle_plants": pl.value(nR[(r,y)]),
                          "new_treatment_plants": pl.value(nT[(r,y)])})
            rowsC.append({"scenario": scenario_name, "region": r, "year": y,
                          "cap_landfill": Lcap, "cap_recycle": Rcap, "cap_treatment": Tcap})
            rowsS.append({"scenario": scenario_name, "region": r, "year": y,
                          "s_unserved": pl.value(sW[(r,y)]),
                          "s_recycle_def": pl.value(sRdef[(r,y)]),
                          "s_treat_def": pl.value(sTdef[(r,y)]),
                          "s_landfill_exc": pl.value(sLexc[(r,y)])})
    total_cost = float(pl.value(pl.lpSum(obj_terms)))
    return (pd.DataFrame(rowsF), pd.DataFrame(rowsN), pd.DataFrame(rowsC),
            total_cost, pd.DataFrame(rowsS))

# -------------------- UI --------------------
st.title("♻️ Türkiye MSW — Karar Destek (Streamlit)")

with st.sidebar:
    st.header("1) Veri")
    up = st.file_uploader("Excel/CSV yükle (opsiyonel)", type=["xlsx","xls","csv"])
    df = read_input_df(up)
    st.caption("Gerekli kolonlar: `region`, `year` ve `waste_*_ton_per_year` (biri yeterli).\
               Kapasiteler varsa: `*_capacity_ton`.")

    st.header("2) Zaman Ufku")
    min_year = int(df["year"].min())
    max_year = max(int(df["year"].max()), min_year+5)
    start_y = st.number_input("Başlangıç yılı", value=min_year, step=1)
    end_y   = st.number_input("Bitiş yılı", value=max_year, step=1)
    YEARS = list(range(int(start_y), int(end_y)+1))

    st.header("3) Senaryo")
    preset = st.selectbox("Ön Ayar", list(DEFAULT_PRESETS.keys()), index=1)
    sp = DEFAULT_PRESETS[preset].copy()
    # kullanıcı ince ayarları
    sp["recycle_target"]    = st.slider("Geri dönüşüm hedefi", 0.0, 0.95, sp["recycle_target"], 0.01)
    sp["min_treat_share"]   = st.slider("Arıtma min. payı",    0.0, 0.50, sp["min_treat_share"], 0.01)
    sp["max_landfill_share"]= st.slider("Depolama max. payı",  0.0, 1.00, sp["max_landfill_share"], 0.01)
    sp["growth_multiplier"] = st.slider("Yıllık büyüme katsayısı", 0.90, 1.10, sp["growth_multiplier"], 0.01)
    sp["uncertainty_pct"]   = st.slider("Belirsizlik (+%)",   0.0, 0.30, sp["uncertainty_pct"], 0.01)
    sp["carbon_price"]      = st.slider("Karbon fiyatı ($/tCO2e)", 0.0, 200.0, sp["carbon_price"], 1.0)

    st.header("4) Emisyon faktörleri (kgCO2e/ton)")
    EMISSION_FACTORS["landfill"]  = st.number_input("Depolama", value=EMISSION_FACTORS["landfill"])
    EMISSION_FACTORS["recycle"]   = st.number_input("Geri dönüşüm", value=EMISSION_FACTORS["recycle"])
    EMISSION_FACTORS["treatment"] = st.number_input("Arıtma", value=EMISSION_FACTORS["treatment"])

    run_btn = st.button("▶︎ Modeli Çalıştır")

st.subheader("Girdi önizleme")
st.dataframe(df.head(10), use_container_width=True)

if run_btn:
    with st.spinner("Çözülüyor..."):
        flows_df, new_df, caps_df, npv_cost, slack_df = solve_multiyear_scenario(df, preset, sp, YEARS)

    # KPI & grafikler
    st.success(f"NPV toplam maliyet: **{npv_cost:,.2f} M$**")

    g = flows_df.groupby("year").agg(
        yL=("y_landfill","sum"), yR=("y_recycle","sum"), yT=("y_treat","sum"), W=("W_used_ton","sum")
    ).reset_index()
    g["Landfill share"] = g["yL"]/g["W"]
    g["Recycle share"]  = g["yR"]/g["W"]
    g["Treatment share"]= g["yT"]/g["W"]

    area_df = g.melt(id_vars="year", value_vars=["Landfill share","Recycle share","Treatment share"],
                     var_name="Flow", value_name="Share")
    fig1 = px.area(area_df, x="year", y="Share", color="Flow",
                   title=f"National Flow Shares — {preset}", groupnorm=None)
    fig1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

    adds = new_df.groupby("year")[["new_landfills","new_recycle_plants","new_treatment_plants"]].sum().reset_index()
    adds = adds.rename(columns={"new_landfills":"Landfill","new_recycle_plants":"Recycle","new_treatment_plants":"Treatment"})
    adds_m = adds.melt(id_vars="year", var_name="Facility", value_name="Count")
    fig2 = px.bar(adds_m, x="year", y="Count", color="Facility", barmode="group",
                  title=f"New Facilities per Year — {preset}")
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Akışlar (ton)")
        st.dataframe(flows_df, use_container_width=True)
        st.download_button("⬇️ Akışlar CSV", data=flows_df.to_csv(index=False).encode("utf-8"),
                           file_name="flows.csv", mime="text/csv")
    with c2:
        st.caption("Yeni tesisler")
        st.dataframe(new_df, use_container_width=True)
        st.download_button("⬇️ Yeni tesisler CSV", data=new_df.to_csv(index=False).encode("utf-8"),
                           file_name="new_facilities.csv", mime="text/csv")

    st.caption("Slack toplamları (ton) — hedef sapmaları")
    st.dataframe(slack_df.groupby("scenario")[["s_unserved","s_recycle_def","s_treat_def","s_landfill_exc"]].sum(),
                 use_container_width=True)
else:
    st.info("Soldan veriyi/parametreleri seçip **Modeli Çalıştır**’a basın.")

st.markdown("---")
st.caption("Not: Kapasiteler veride yoksa başlangıç payları ve `BASELINE_HEADROOM` ile türetilir. \
Integer tesis sayısı gerekirse kod başındaki `USE_INTEGER_FACILITIES=True` yapabilirsiniz (CBC).")

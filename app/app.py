# app.py

import re, json
from functools import lru_cache
from typing import Iterable, Optional
import pandas as pd
import pulp as pl
import streamlit as st
from io import StringIO

CSV_PATH = "./app/goods.csv"

# ----------------------------
# Page config + CSS
# ----------------------------
st.set_page_config(page_title="Hay Day XP Optimizer", page_icon=":ear_of_rice:", layout="wide")
st.markdown("""
<style>
:root {
  --butter: #FFF4C7;
  --butter-2: #FFEFA8;
  --ink: #2F2A1E;
  --white: #FFFFFF;
  --accent: #FFB703;
  --accent-2: #FFD166;
  --line: #EADDA6;
  --input-border: #D8CBA0;
}

/* Base Style */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--butter);
  color: var(--ink);
}

/* Header and Sidebar */
[data-testid="stHeader"] {
  background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #FFF7D9, #FFF1B8);
  border-right: 1px solid var(--line);
}

/* Headings */
h1, h2, h3, h4 { font-family: 'DM Serif Display', system-ui; font-weight: 700; letter-spacing: 0.2px; }
h1 { color: #3A2F16;} 

/* Input Fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div {
  border-radius: 12px;
  border: 1px solid var(--input-border);
  background-color: var(--white);
  color: var(--ink);
}

/* Tables/Dataframes */
.tbl {
  background: #fffceb;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 8px 8px 2px;
  box-shadow: 0 8px 18px rgba(0,0,0,0.04);
}

/* Buttons */
div.stButton > button[kind="primary"] {
  background: var(--accent);
  color: #231F0F;
  border: 0;
  border-radius: 999px;
  padding: 0.6rem 1.1rem;
  font-weight: 700;
}
div.stButton > button[kind="primary"]:hover {
  background: var(--accent-2);
}
</style>
""", unsafe_allow_html=True)

st.title(":ear_of_rice::chicken::tractor: Hay Day XP Optimizer")
st.caption("Input your level and time window and I will make a plan for you!")

# ----------------------------
# Parsers & Helpers
# ----------------------------

def norm(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s)
    s = s.replace("\u00a0"," ").replace("–","-").replace("—","-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def finals_in_unlocked(av: pd.DataFrame) -> set:
    items = set(av["Name"])
    used = set()
    for _, r in av.iterrows():
        # r['needs_norm'] contains the normalized name keys used for consumption
        # We check if the ingredient is available in the current unlocked items
        for ing_norm in r["needs_norm"].keys():
            # Get the original name for the ingredient (approximation, but safe if df is properly normalized)
            ing_original_name = df[df['name_norm'] == ing_norm]['Name'].iloc[0] if ing_norm in df['name_norm'].values else None
            if ing_original_name in items:
                used.add(ing_original_name)
    return items - used

@st.cache_data(show_spinner=False)
def load_final_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Needs_norm is stored as a stringified dict in CSV, so we must parse it back
    # The 'Needs' column needs to be parsed before being used by the solver
    def safe_literal_eval(x):
        try:
            # Safely converts string representation of dict/list back to Python object
            if isinstance(x, str):
                x = x.replace("'", '"') # Adjust single quotes to double quotes for json.loads
                return json.loads(x)
            return {}
        except Exception:
            return {}
    
    df['needs_norm'] = df['needs_norm'].apply(safe_literal_eval)
    
    return df

# ----------------------------
# Optimizer
# ----------------------------

def optimize_plan_notebook(
    df_cleaned: pd.DataFrame,
    player_level: int,
    T_hours: float = 24.0,
    integer_solution: bool = True 
) -> dict:
    
    avail = df_cleaned[df_cleaned["Level_num"] <= player_level].copy()
    total_cap_min = max(0.0, T_hours * 60.0)

    # Pre-calculated maps
    items_set = set(avail["name_norm"])
    xp        = {r["name_norm"]: float(r["XP"]) for _, r in avail.iterrows()}
    tmin      = {r["name_norm"]: float(r["time_min"])  for _, r in avail.iterrows()}
    needs_map = {r["name_norm"]: dict(r["needs_norm"]) for _, r in avail.iterrows()}
    yield_map = {r["name_norm"]: float(r["Yield_qty"]) for _, r in avail.iterrows()}
    machine_map = {r["name_norm"]: r["Building"] for _, r in avail.iterrows()}
    type_map = {r["name_norm"]: r["Production_Type"] for _, r in avail.iterrows()}
    canon_name = df_cleaned.groupby("name_norm")["Name"].first().to_dict()

    # Initial Stock Setup (30.0 for crops/trees/bushes)
    initial_stock_map = {}
    CROP_SOURCE_PATTERNS = r'Field|Tree|Bush' 
    for _, r in avail.iterrows():
        if re.search(CROP_SOURCE_PATTERNS, str(r.get('Source', '')), re.I):
            initial_stock_map[r['name_norm']] = 30.0 

    # Feasibility Check (Closure)
    @lru_cache(None)
    def closure_ok(u: str) -> bool:
        for ing, q in needs_map.get(u, {}).items():
            if ing not in items_set: return False
            if not closure_ok(ing): return False
        return True

    finals_all = finals_in_unlocked(avail)
    finals_raw = {norm(i) for i in finals_all}
    finals = {i for i in finals_raw if closure_ok(i)}
    
    ingredient_items = set()
    for nd in needs_map.values():
        for k in nd.keys():
            if k in items_set: ingredient_items.add(k)

    # Build MILP
    model = pl.LpProblem("HayDay_XP_Max_Notebook", pl.LpMaximize)
    v = lambda p, s: f"{p}{re.sub(r'[^A-Za-z0-9]+','_', s)}"
    
    cat = "Integer" if integer_solution else "Continuous"
    x = {i: pl.LpVariable(v("x", i), lowBound=0, cat=cat) for i in items_set}

    # GOAL: MAXIMIZE XP!!!
    model+=pl.lpSum(xp[i] * x[i] for i in items_set)
            
    # Ingredient Constraints
    for k in ingredient_items:
        consume=pl.lpSum(needs_map[i].get(k, 0.0) * x[i] for i in items_set)
        initial_stock=initial_stock_map.get(k, 0.0)
        
        if k in finals or initial_stock > 0:
            # Case 1: Final output or has stock (Must produce at least consumption)
            model+=yield_map.get(k, 1.0) * x[k] >= (consume - initial_stock), v("balance_ge", k)
        else:
            # Case 2: Pure ingredient, no stock (Must produce exact consumption)
            model+= ield_map.get(k, 1.0) * x[k] == consume, v("balance_eq", k)
    
    # Machine Constraint
    active_buildings = {r["Building"] for _, r in avail.iterrows()}
    MACHINE_CAP_MIN = 1440.0 # 24hrs * 60min/hour
    
    for machine in active_buildings:
        machine_time = pl.lpSum(
            tmin[i] * x[i]
            for i in items_set
            if avail[avail['name_norm'] == i]['Building'].iloc[0] == machine
        )
        model += machine_time <= MACHINE_CAP_MIN, f"machine_cap_{machine}"
        
    # Global Time Cap
    model += pl.lpSum(tmin[i] * x[i] for i in items_set) <= total_cap_min, "global_time_cap"
        
    # Solve and Extract
    model.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[model.status]
    
    # Define default failure return 
    if status != "Optimal":
        total_cap_min = T_hours * 60.0
        cols = ["item","is_final","qty","xp_each","xp_total","time_min_each","time_min_total"]
        return {
            "status": status,
            "XP_total": 0.0,
            "XP_per_hour": 0.0,
            "total_time_used_min": 0.0,
            "time_capacity_min": total_cap_min,
            "plan": pd.DataFrame(columns=cols)
        }

    # Optimal Extraction
    rows, total_xp, total_time_used = [], 0.0, 0.0
    
    for i in items_set:
        val = x[i].value()
        if val is None: continue
        
        qty = int(round(val)) if integer_solution else float(val)
        if qty > 1e-6:
            
            display_name = df_cleaned[df_cleaned['name_norm'] == i]['Name'].iloc[0]
            is_final = i in finals
            
            rows.append({
                "item": display_name,
                "is_final": is_final,
                "qty": qty,
                "xp_each": xp[i],
                "xp_total": xp[i] * qty,
                "time_min_each": tmin[i],
                "time_min_total": tmin[i] * qty
            })
            total_xp += xp[i] * qty
            total_time_used += tmin[i] * qty

    plan_df = pd.DataFrame(rows)
    
    # FINAL FILTER: SHOW ONLY MACHINE/PROCESSED GOODS!!
    plan_df['production_type'] = plan_df['item'].apply(
        lambda x: type_map.get(norm(x), 'Unknown')
    )

    plan_df = plan_df[plan_df['production_type'] == 'Machine/Processed'].drop(
        columns=['production_type']
    )
    
    plan_df = plan_df.sort_values(
        ["is_final", "xp_each", "time_min_each"], 
        ascending=[False, False, True]
    ).reset_index(drop=True)
    # ========================================================
    
    return {
        "status": status,
        "XP_total": total_xp,
        "XP_per_hour": (total_xp / T_hours) if T_hours > 0 else float("nan"),
        "total_time_used_min": total_time_used,
        "time_capacity_min": total_cap_min,
        "plan": plan_df
    }

# ----------------------------
# Load data
# ----------------------------
try:
    df = load_final_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load dataset at {CSV_PATH}. Have you run your notebook to create goods_final.csv? Error: {e}")
    st.stop()

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header(":gear: Settings")

player_level = st.sidebar.number_input("Player level", min_value=1, max_value=500, value=72, step=1)
time_window  = st.sidebar.number_input("Time window (hours)", min_value=0.25, max_value=168.0, value=24.0, step=0.25)
# Set integer solution to True by default, as it provides stable MILP solution
integer_sol  = st.sidebar.checkbox("Whole items only (integer solution)", value=True) 

st.sidebar.markdown("---")
st.sidebar.subheader("Display")

unlocked = df[df["Level_num"] <= player_level].copy()
finals_set = finals_in_unlocked(unlocked)
show_only_finals = st.sidebar.checkbox("Show only final products in table", value=False)

sort_choice = st.sidebar.selectbox(
    "Sort table by",
    ["XP each (desc), Time each (asc)", "XP total (desc)", "Time total (desc)"],
    index=0
)

run_btn = st.sidebar.button("Optimize :carrot:", type="primary")

# ----------------------------
# Run + render
# ----------------------------
if run_btn:
    with st.spinner("Solving…"):
        res = optimize_plan_notebook(
            df, 
            player_level=player_level,
            T_hours=time_window,
            integer_solution=integer_sol
        )

    # KPIs (Box-free layout)
    k1, k2, k3, k4 = st.columns(4)
    
    used = res["total_time_used_min"]
    cap = res["time_capacity_min"]
    plan = res["plan"]
    
    k1.metric("Total XP", f"{res['XP_total']:,.0f}")
    k2.metric("XP / hour", f"{res['XP_per_hour']:,.1f}")
    k3.metric("Time used", f"{used:,.1f} / {cap:,.1f} min")
    k4.metric("Items in plan", 0 if plan.empty else len(plan))

    st.markdown("### Plan")

    if res['status'] != "Optimal":
         st.error(f"Solver Status: **{res['status']}**. Could not find an optimal solution. Try increasing the time window or lowering the level.")
    elif plan.empty:
        st.warning("No machine-made items selected. Try increasing hours or level, or check if all necessary ingredients are unlocked.", icon=":warning:")
    else:
        # Filtering and Sorting
        view = plan.copy()
        if show_only_finals:
            view = view[view["item"].isin(finals_set)].copy()

        if sort_choice == "XP total (desc)":
            view = view.sort_values("xp_total", ascending=False)
        elif sort_choice == "Time total (desc)":
            view = view.sort_values("time_min_total", ascending=False)
        else:
            view = view.sort_values(["is_final", "xp_each","time_min_each"], ascending=[False, False, True])

        st.markdown('<div class="tbl">', unsafe_allow_html=True)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Charts
        chart_base = view.loc[view["xp_total"] > 0] 
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("XP by Item")
            st.bar_chart(chart_base.set_index("item")["xp_total"])
        with c2:
            st.subheader("Time by Item (min)")
            st.bar_chart(chart_base.set_index("item")["time_min_total"])
            
        # Download plan
        csv_buf = StringIO()
        view.to_csv(csv_buf, index=False)
        st.download_button(
            "Download plan as CSV",
            csv_buf.getvalue(),
            file_name="hayday_plan.csv",
            mime="text/csv"
        )
else:
    st.info("Pick your level & time, then click **Optimize :carrot:**")
###############################################################
#   GymIQ - Inventory & Revenue Intelligence for Gyms
#   PART A: Imports, Config, Sample Data Generator
###############################################################

import os
import random
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="GymIQ – Inventory & Revenue Intelligence for Gyms",
    layout="wide",
)

# ---- Access code gate ----
ACCESS_CODE = "GYM-TEST"     # change for real gym owners

# ---- Data file path ----
DATA_PATH = Path("data/gymIQ_sample.csv")

# ---- Column mappings (same schema as AlcIQ so engine works) ----
DATE_COL = "date"                     
SKU_COL = "sku"
NAME_COL = "product_name"
CAT_COL = "category"
SUPPLIER_COL = "supplier"
UNITS_COL = "units_sold"
REV_COL = "revenue"
INV_COL = "inventory_on_hand"
COST_COL = "unit_cost"
LEADTIME_COL = "lead_time_days"

# ---- Default tuning values ----
DEFAULT_HISTORY_DAYS = 90
DEFAULT_FORECAST_DAYS = 14
DEFAULT_SAFETY_FACTOR = 0.5
DEFAULT_TARGET_SERVICE_DAYS = 21
DEFAULT_MIN_SLOW_DAILY_UNITS = 0.02
DEFAULT_FAST_TOP_N = 15
DEFAULT_SLOW_TOP_N = 15

# Initialize session state
if "history_days" not in st.session_state:
    st.session_state["history_days"] = DEFAULT_HISTORY_DAYS
if "forecast_days" not in st.session_state:
    st.session_state["forecast_days"] = DEFAULT_FORECAST_DAYS
if "safety_factor" not in st.session_state:
    st.session_state["safety_factor"] = DEFAULT_SAFETY_FACTOR
if "target_service_days" not in st.session_state:
    st.session_state["target_service_days"] = DEFAULT_TARGET_SERVICE_DAYS
if "min_slow_daily_units" not in st.session_state:
    st.session_state["min_slow_daily_units"] = DEFAULT_MIN_SLOW_DAILY_UNITS
if "fast_top_n" not in st.session_state:
    st.session_state["fast_top_n"] = DEFAULT_FAST_TOP_N
if "slow_top_n" not in st.session_state:
    st.session_state["slow_top_n"] = DEFAULT_SLOW_TOP_N


# ============================================================
# SAMPLE DATA GENERATOR (GYM / STUDIO)
# ============================================================

def generate_sample_dataset(
    path: Path,
    n_skus: int = 220,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
):
    """
    Generates a realistic Gym dataset covering:
    - Retail: energy drinks, preworkout, protein, snacks, apparel, accessories
    - Services: PT packages, class packs, day passes
    - Seasonality: Jan spikes, summer supplement surges, weekend check-ins
    - Lead times and reorder logic for retail items
    """
    rng = np.random.default_rng(42)
    random.seed(42)

    dates = pd.date_range(start_date, end_date, freq="D")

    # Gym-specific categories
    categories = [
        "Energy Drinks",
        "Protein Powder",
        "Preworkout",
        "Bars & Snacks",
        "Apparel",
        "Accessories",
        "PT Packages",
        "Class Packs",
        "Day Passes",
    ]

    cat_weights = np.array([
        0.20, 0.12, 0.10, 0.10,
        0.10, 0.08, 0.12, 0.10, 0.08
    ])
    cat_weights /= cat_weights.sum()

    suppliers_by_cat = {
        "Energy Drinks": ["Celsius", "RedBull", "Monster", "Bang"],
        "Protein Powder": ["Optimum", "Dymatize", "Ghost"],
        "Preworkout": ["C4", "AlaniNu", "Legion"],
        "Bars & Snacks": ["Quest", "CLIF", "RXBar"],
        "Apparel": ["GymIQ Apparel", "LocalPrintCo"],
        "Accessories": ["Rogue", "GymShark", "GenericFitness"],
        "PT Packages": ["In-House Coaches"],
        "Class Packs": ["Studio Classes"],
        "Day Passes": ["Front Desk"],
    }

    base_demand_map = {
        "Energy Drinks": (1.5, 4.0),
        "Protein Powder": (0.2, 0.8),
        "Preworkout": (0.3, 1.2),
        "Bars & Snacks": (0.8, 2.0),
        "Apparel": (0.05, 0.2),
        "Accessories": (0.05, 0.2),
        "PT Packages": (0.02, 0.08),
        "Class Packs": (0.05, 0.15),
        "Day Passes": (0.2, 0.8),
    }

    cost_map = {
        "Energy Drinks": (1.0, 1.4),
        "Protein Powder": (18, 35),
        "Preworkout": (12, 25),
        "Bars & Snacks": (0.8, 1.5),
        "Apparel": (8, 20),
        "Accessories": (5, 25),
        "PT Packages": (15, 35),   # cost is coach payout or marginal cost
        "Class Packs": (5, 15),
        "Day Passes": (3, 6),
    }

    lead_time_map = {
        "Energy Drinks": (5, 10),
        "Protein Powder": (7, 14),
        "Preworkout": (7, 14),
        "Bars & Snacks": (5, 10),
        "Apparel": (14, 30),
        "Accessories": (10, 21),
        "PT Packages": (0, 0),
        "Class Packs": (0, 0),
        "Day Passes": (0, 0),
    }

    sku_rows = []
    for i in range(n_skus):
        sku_id = f"SKU-{i+1:04d}"
        cat = rng.choice(categories, p=cat_weights)
        supplier = random.choice(suppliers_by_cat[cat])

        lam_low, lam_high = base_demand_map.get(cat, (0.05, 0.2))
        c_low, c_high = cost_map.get(cat, (5, 20))
        unit_cost = rng.uniform(c_low, c_high)

        # Higher margin for services
        if cat in ["PT Packages", "Class Packs", "Day Passes"]:
            margin_pct = rng.uniform(0.55, 0.75)
        else:
            margin_pct = rng.uniform(0.30, 0.45)

        unit_price = unit_cost * (1 + margin_pct)

        lt_low, lt_high = lead_time_map.get(cat, (7, 14))
        lead_time_days = int(rng.integers(lt_low, lt_high + 1))

        sku_rows.append(
            {
                SKU_COL: sku_id,
                CAT_COL: cat,
                SUPPLIER_COL: supplier,
                COST_COL: unit_cost,
                "unit_price": unit_price,
                LEADTIME_COL: lead_time_days,
                "lam_low": lam_low,
                "lam_high": lam_high,
            }
        )

    sku_df = pd.DataFrame(sku_rows)

    # Map product names
    name_by_cat = {
        "Energy Drinks": ["Energy Drink 16oz", "Sugar-Free Energy 16oz"],
        "Protein Powder": ["Whey Protein 2lb", "Plant Protein 2lb"],
        "Preworkout": ["Preworkout Tub", "Stim-Free Pre"],
        "Bars & Snacks": ["Protein Bar", "Granola Bar", "Crispy Bar"],
        "Apparel": ["Gym Tee", "Hoodie", "Training Shorts"],
        "Accessories": ["Lifting Belt", "Wrist Wraps", "Shaker Bottle"],
        "PT Packages": ["5 PT Sessions", "10 PT Sessions"],
        "Class Packs": ["10-Class Pack", "20-Class Pack"],
        "Day Passes": ["Day Pass", "Weekend Pass"],
    }

    def make_name(cat: str) -> str:
        return random.choice(name_by_cat.get(cat, ["Gym Product"]))

    sku_df[NAME_COL] = sku_df[CAT_COL].apply(make_name)

    # Build daily data rows
    rows = []
    for _, sku_row in sku_df.iterrows():

        sku_id = sku_row[SKU_COL]
        cat = sku_row[CAT_COL]
        supplier = sku_row[SUPPLIER_COL]
        unit_cost = sku_row[COST_COL]
        unit_price = sku_row["unit_price"]
        lead_time = sku_row[LEADTIME_COL]
        product_name = sku_row[NAME_COL]

        lam_low = sku_row["lam_low"]
        lam_high = sku_row["lam_high"]

        base_lambda = rng.uniform(lam_low, lam_high)

        # Inventory only for retail items
        inventory = int(rng.integers(10, 80)) if lead_time > 0 else 0
        reorder_point = int(inventory * 0.4)
        reorder_qty = int(inventory * 0.8)

        for d in dates:
            month = d.month
            dow = d.weekday()

            demand_factor = 1.0

            # January membership spike
            if month in [1, 2]:
                if cat in ["PT Packages", "Class Packs", "Day Passes"]:
                    demand_factor *= 1.7
                else:
                    demand_factor *= 1.25

            # Summer supplement spike
            if month in [5, 6, 7]:
                if cat in ["Energy Drinks", "Preworkout", "Protein Powder"]:
                    demand_factor *= 1.5

            # Weekend check-in spike
            if dow in [5, 6]:
                if cat in ["Day Passes", "Class Packs", "Energy Drinks"]:
                    demand_factor *= 1.4

            lam = max(base_lambda * demand_factor, 0.01)

            units_sold = int(rng.poisson(lam))

            # Inventory-based categories
            if lead_time > 0:
                units_sold = min(units_sold, inventory)
                inventory -= units_sold

                # auto reorder
                if inventory < reorder_point:
                    inventory += reorder_qty + rng.integers(5, 20)
            else:
                # service
                inventory = 0

            revenue = units_sold * unit_price

            rows.append(
                {
                    DATE_COL: d,
                    SKU_COL: sku_id,
                    NAME_COL: product_name,
                    CAT_COL: cat,
                    SUPPLIER_COL: supplier,
                    UNITS_COL: units_sold,
                    REV_COL: revenue,
                    INV_COL: inventory,
                    COST_COL: unit_cost,
                    LEADTIME_COL: lead_time,
                }
            )

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
###############################################################
#   GymIQ - PART B: Analytics, Metrics, Utilities
###############################################################

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading gym sales & inventory data…")
def load_data(path: Path) -> pd.DataFrame:
    """Read the user-uploaded or sample dataset with full validation."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path.resolve()}")

    df = pd.read_csv(path)

    # Required structure for GymIQ
    required_cols = [
        DATE_COL,
        SKU_COL,
        NAME_COL,
        CAT_COL,
        SUPPLIER_COL,
        UNITS_COL,
        REV_COL,
        INV_COL,
        COST_COL,
        LEADTIME_COL,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce data types
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    numeric_cols = [UNITS_COL, REV_COL, INV_COL, COST_COL, LEADTIME_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def get_file_last_updated(path: Path) -> str:
    """Human-readable timestamp for display."""
    try:
        ts = datetime.fromtimestamp(path.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown"


def get_date_range(df: pd.DataFrame) -> Tuple[datetime, datetime]:
    return df[DATE_COL].min(), df[DATE_COL].max()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

def add_sidebar_settings():
    """User controls for forecast horizon, safety stock, etc."""
    st.sidebar.markdown("## ⚙️ Global Settings for GymIQ")
    st.sidebar.caption(
        "These settings control how GymIQ measures demand, risk, and "
        "the recommended order quantities. Defaults are safe for most gyms."
    )

    st.session_state["history_days"] = st.sidebar.slider(
        "📘 Sales history window (days)",
        30, 365,
        value=st.session_state["history_days"],
        step=15,
    )
    st.session_state["forecast_days"] = st.sidebar.slider(
        "📅 Forecast horizon (days)",
        7, 60,
        value=st.session_state["forecast_days"],
        step=7,
    )
    st.session_state["safety_factor"] = st.sidebar.slider(
        "🛡️ Safety stock factor",
        0.0, 1.5,
        value=float(st.session_state["safety_factor"]),
        step=0.1,
    )
    st.session_state["target_service_days"] = st.sidebar.slider(
        "🎯 Target service days",
        7, 60,
        value=st.session_state["target_service_days"],
        step=7,
    )
    st.session_state["min_slow_daily_units"] = st.sidebar.number_input(
        "🐌 Minimum daily units for slow-mover rankings",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["min_slow_daily_units"]),
        step=0.01,
    )
    st.session_state["fast_top_n"] = st.sidebar.number_input(
        "🚀 Top N fast movers",
        5, 50,
        value=int(st.session_state["fast_top_n"]),
        step=5,
    )
    st.session_state["slow_top_n"] = st.sidebar.number_input(
        "🐢 Top N slow movers",
        5, 50,
        value=int(st.session_state["slow_top_n"]),
        step=5,
    )


# ============================================================
# TEMPLATE FILE (EMPTY)
# ============================================================

def make_template_df() -> pd.DataFrame:
    """Structure for file upload template."""
    cols = [
        DATE_COL, SKU_COL, NAME_COL, CAT_COL, SUPPLIER_COL,
        UNITS_COL, REV_COL, INV_COL, COST_COL, LEADTIME_COL
    ]
    return pd.DataFrame(columns=cols)


# ============================================================
# CORE INVENTORY METRICS ENGINE
# ============================================================

def compute_inventory_metrics(
    df: pd.DataFrame,
    history_days: int,
    forecast_days: int,
    safety_factor: float,
    target_service_days: int,
) -> pd.DataFrame:
    """
    Compute:
    - avg daily units
    - safety stock
    - inventory value
    - projected balance
    - recommended orders
    - weeks on hand
    - ABC revenue class (A/B/C)
    - Inventory health status
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    max_date = df[DATE_COL].max()
    hist_start = max_date - timedelta(days=history_days)

    recent = df[df[DATE_COL] >= hist_start]

    # Average daily units sold
    daily_units = recent.groupby(SKU_COL)[UNITS_COL].sum() / max(history_days, 1)

    # Latest inventory
    current_inventory = (
        df.sort_values(DATE_COL).groupby(SKU_COL)[INV_COL].last()
    )

    # Base info
    base = (
        df.sort_values(DATE_COL)
        .groupby(SKU_COL)
        .agg({
            NAME_COL: "last",
            CAT_COL: "last",
            SUPPLIER_COL: "last",
            COST_COL: "last",
            LEADTIME_COL: "last",
        })
    )

    metrics = base.copy()

    metrics["avg_daily_units"] = daily_units.fillna(0)
    metrics["current_inventory"] = current_inventory.fillna(0)
    metrics["forecast_demand"] = metrics["avg_daily_units"] * forecast_days

    # Handle zero lead times (services)
    metrics[LEADTIME_COL] = metrics[LEADTIME_COL].replace(0, np.nan).fillna(7)

    # Safety stock = (daily units × lead time × safety factor)
    metrics["safety_stock"] = (
        metrics["avg_daily_units"] * metrics[LEADTIME_COL] * safety_factor
    )

    # Weeks on hand
    metrics["weeks_on_hand"] = np.where(
        metrics["avg_daily_units"] > 0,
        metrics["current_inventory"] / (metrics["avg_daily_units"] * 7),
        np.inf,
    )

    metrics["inventory_value"] = metrics["current_inventory"] * metrics[COST_COL]
    metrics["dead_inventory_value"] = np.where(
        metrics["avg_daily_units"] < 0.01,
        metrics["inventory_value"],
        0
    )

    metrics["projected_balance"] = (
        metrics["current_inventory"]
        - metrics["forecast_demand"]
        + metrics["safety_stock"]
    )

    # Status classification
    stat_list = []
    for _, r in metrics.iterrows():
        inv = r["current_inventory"]
        demand = r["forecast_demand"]
        pb = r["projected_balance"]

        if inv <= 0 or pb < 0:
            stat_list.append("🔥 Stockout Risk")
        elif inv < demand:
            stat_list.append("🟡 Low Inventory")
        elif inv > demand * 2:
            stat_list.append("🔵 Overstock")
        else:
            stat_list.append("✅ Healthy")

    metrics["status"] = stat_list

    # Target inventory
    target_days = forecast_days + metrics[LEADTIME_COL] + target_service_days
    metrics["target_inventory"] = (
        metrics["avg_daily_units"] * target_days + metrics["safety_stock"]
    )

    # Recommended orders
    metrics["recommended_order_qty"] = np.where(
        metrics["status"].isin(["🔥 Stockout Risk", "🟡 Low Inventory"]),
        np.maximum(metrics["target_inventory"] - metrics["current_inventory"], 0),
        0,
    ).round().astype(int)

    # ABC revenue classification
    recent_rev = recent.groupby(SKU_COL)[REV_COL].sum()
    total_rev = recent_rev.sum()

    share = recent_rev / total_rev if total_rev > 0 else recent_rev * 0
    sorted_share = share.sort_values(ascending=False)
    cum = sorted_share.cumsum()

    abc_map = {}
    for sku, cs in cum.items():
        if cs <= 0.8:
            abc_map[sku] = "A"
        elif cs <= 0.95:
            abc_map[sku] = "B"
        else:
            abc_map[sku] = "C"

    metrics["abc_class"] = metrics.index.map(lambda x: abc_map.get(x, "C"))

    metrics.reset_index(inplace=True)
    return metrics


# ============================================================
# SEASONALITY
# ============================================================

def compute_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Month-to-month volatility for each SKU."""
    if df.empty:
        return pd.DataFrame()

    temp = df.copy()
    temp["month"] = temp[DATE_COL].dt.month

    monthly = temp.groupby([SKU_COL, "month"])[UNITS_COL].sum().reset_index()

    summary = monthly.groupby(SKU_COL)[UNITS_COL].agg(["mean", "std"]).reset_index()
    summary["seasonality_score"] = np.where(
        summary["mean"] > 0,
        summary["std"] / summary["mean"],
        0,
    )

    return summary[[SKU_COL, "seasonality_score"]]


# ============================================================
# CATEGORY SUMMARY
# ============================================================

def compute_category_summary(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if df.empty or metrics.empty:
        return pd.DataFrame()

    max_date = df[DATE_COL].max()
    hist_start = max_date - timedelta(days=st.session_state["history_days"])

    recent = df[df[DATE_COL] >= hist_start]

    cat_rev = recent.groupby(CAT_COL)[REV_COL].sum()
    cat_units = recent.groupby(CAT_COL)[UNITS_COL].sum()
    cat_inv_val = metrics.groupby(CAT_COL)["inventory_value"].sum()
    cat_dead_val = metrics.groupby(CAT_COL)["dead_inventory_value"].sum()

    # Normalize
    total_rev = cat_rev.sum()
    total_units = cat_units.sum()
    total_inv = cat_inv_val.sum()

    df_cat = pd.DataFrame({
        CAT_COL: cat_rev.index,
        "revenue": cat_rev.values,
        "units": cat_units.values,
        "inventory_value": cat_inv_val.values,
        "dead_inventory_value": cat_dead_val.values,
    })

    df_cat["revenue_share"] = df_cat["revenue"] / total_rev if total_rev > 0 else 0
    df_cat["units_share"] = df_cat["units"] / total_units if total_units > 0 else 0
    df_cat["inventory_share"] = df_cat["inventory_value"] / total_inv if total_inv > 0 else 0

    return df_cat.sort_values("revenue", ascending=False)


# ============================================================
# FAST & SLOW MOVERS
# ============================================================

def get_slow_fast_movers(
    metrics: pd.DataFrame,
    slow_n: int,
    fast_n: int,
    min_slow_daily_units: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()

    slow = (
        metrics[metrics["avg_daily_units"] >= min_slow_daily_units]
        .sort_values("avg_daily_units", ascending=True)
        .head(slow_n)
    )

    fast = (
        metrics.sort_values("avg_daily_units", ascending=False)
        .head(fast_n)
    )

    return slow, fast
###############################################################
#   GymIQ - PART C: Pages + Main App Runner
###############################################################

# ============================================================
# PAGE: DATA UPLOAD FIRST
# ============================================================

def page_data_upload():
    st.header("📁 Upload Your Gym Data")

    st.markdown(
        """
        GymIQ starts by analyzing **your sales, memberships, PT sessions, class packs, retail items, and inventory**.  
        Upload a file that matches the GymIQ template to begin.

        ### 💡 Recommended:
        Start with the **CSV template**, fill it out with your gym's daily data, then upload it below.
        """
    )

    # Download CSV template
    tmpl_df = make_template_df()
    csv_data = tmpl_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download GymIQ CSV Template",
        data=csv_data,
        file_name="gymiq_template.csv",
        mime="text/csv",
    )

    # Excel template (best effort)
    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            tmpl_df.to_excel(writer, index=False, sheet_name="Template")
        buf.seek(0)

        st.download_button(
            label="⬇️ Download Excel Template (.xlsx)",
            data=buf,
            file_name="gymiq_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.info("Excel template not available in this environment. Use the CSV template above.")

    st.markdown("---")
    st.subheader("📤 Upload Your Data File")

    uploaded = st.file_uploader(
        "Upload your CSV or Excel file here",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                new_df = pd.read_csv(uploaded)
            else:
                new_df = pd.read_excel(uploaded)

            # Validate columns
            required = make_template_df().columns
            missing = [c for c in required if c not in new_df.columns]
            if missing:
                st.error(f"Your file is missing required columns: {missing}")
                return

            # Save to data path
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(DATA_PATH, index=False)

            st.success("Your data has been uploaded and saved. GymIQ is now using your file.")
            st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")


# ============================================================
# PAGE: OVERVIEW DASHBOARD
# ============================================================

def page_overview(df: pd.DataFrame, metrics: pd.DataFrame):
    st.header("📊 GymIQ Overview Dashboard")

    st.markdown(
        """
        This is your **gym’s performance snapshot** across retail items, PT packages, class packs, day passes, 
        and overall sales.

        You'll see:
        - Top product categories  
        - Inventory health status  
        - Revenue drivers  
        - Risk areas (low stock or dead inventory)
        """
    )

    # Date range
    min_date, max_date = get_date_range(df)
    st.caption(f"Data from **{min_date.date()}** to **{max_date.date()}**")

    st.subheader("🏷️ Category Performance")
    cat_df = compute_category_summary(df, metrics)
    st.dataframe(cat_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 Inventory Health Breakdown")

    status_counts = metrics["status"].value_counts().rename_axis("status").reset_index(name="count")
    st.dataframe(status_counts, use_container_width=True)


# ============================================================
# PAGE: FAST & SLOW MOVERS
# ============================================================

def page_fast_slow(metrics: pd.DataFrame):
    st.header("🚀 Fast Movers & 🐢 Slow Movers")

    st.markdown(
        """
        Fast movers = products or services selling **quickly and consistently**  
        Slow movers = products that **barely move**, often tying up inventory or space
        """
    )

    slow, fast = get_slow_fast_movers(
        metrics,
        st.session_state["slow_top_n"],
        st.session_state["fast_top_n"],
        st.session_state["min_slow_daily_units"],
    )

    st.subheader("🐢 Slow Movers")
    st.dataframe(slow, use_container_width=True)

    st.markdown("---")
    st.subheader("🚀 Fast Movers")
    st.dataframe(fast, use_container_width=True)


# ============================================================
# PAGE: PURCHASE ORDERS
# ============================================================

def page_purchase_orders(metrics: pd.DataFrame):
    st.header("📦 Recommended Purchase Orders")

    st.markdown(
        """
        GymIQ calculates exactly **how much you should reorder** based on:
        - Recent demand  
        - Forecast window  
        - Lead time  
        - Safety stock buffer  
        - Desired days of service coverage  

        Only items with **Stockout Risk** or **Low Inventory** show up here.
        """
    )

    po_df = metrics[metrics["recommended_order_qty"] > 0].copy()

    if po_df.empty:
        st.success("🎉 You’re fully stocked! No purchase orders needed.")
        return

    st.dataframe(
        po_df[
            [SKU_COL, NAME_COL, CAT_COL, SUPPLIER_COL,
             "current_inventory", "forecast_demand", "recommended_order_qty"]
        ],
        use_container_width=True,
    )


# ============================================================
# PAGE: SKU EXPLORER
# ============================================================

def page_sku_explorer(df: pd.DataFrame, metrics: pd.DataFrame):
    st.header("🔍 SKU Explorer")

    st.markdown(
        """
        Search any product or service to see:
        - Its demand  
        - Inventory levels  
        - Seasonality  
        - Recommended reorder quantities  
        """
    )

    sku_list = metrics[SKU_COL].unique()
    sku = st.selectbox("Select a SKU", sku_list)

    row_m = metrics[metrics[SKU_COL] == sku].iloc[0]
    row_df = df[df[SKU_COL] == sku]

    st.subheader(f"📌 {row_m[NAME_COL]}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Units", f"{row_m['avg_daily_units']:.3f}")
    col2.metric("Weeks on Hand", f"{row_m['weeks_on_hand']:.2f}")
    col3.metric("Status", row_m["status"])

    st.markdown("---")
    st.subheader("📅 Sales History")
    st.line_chart(row_df.set_index(DATE_COL)[UNITS_COL])


# ============================================================
# PAGE: SEASONALITY
# ============================================================

def page_seasonality(df: pd.DataFrame):
    st.header("📆 Seasonality Insights")

    st.markdown(
        """
        GymIQ analyzes **month-to-month volatility** for each SKU or service.  
        A higher score means the product sells much more in certain months 
        (example: January PT Packages or summer supplements).
        """
    )

    seas = compute_seasonality(df)
    st.dataframe(seas.sort_values("seasonality_score", ascending=False), use_container_width=True)


# ============================================================
# PAGE: SETTINGS
# ============================================================

def page_settings():
    st.header("⚙️ Global Settings")

    st.markdown(
        """
        Adjust how GymIQ measures demand and recommends stock.  
        These values affect **all dashboards and reorder suggestions**.
        """
    )

    add_sidebar_settings()

    if st.button("Reset to Defaults"):
        st.session_state["history_days"] = DEFAULT_HISTORY_DAYS
        st.session_state["forecast_days"] = DEFAULT_FORECAST_DAYS
        st.session_state["safety_factor"] = DEFAULT_SAFETY_FACTOR
        st.session_state["target_service_days"] = DEFAULT_TARGET_SERVICE_DAYS
        st.session_state["min_slow_daily_units"] = DEFAULT_MIN_SLOW_DAILY_UNITS
        st.session_state["fast_top_n"] = DEFAULT_FAST_TOP_N
        st.session_state["slow_top_n"] = DEFAULT_SLOW_TOP_N
        st.success("Settings reset.")
        st.rerun()


# ============================================================
# PAGE: RAW DATA
# ============================================================

def page_raw_data(df: pd.DataFrame):
    st.header("📄 Raw Data")
    st.dataframe(df, use_container_width=True)


# ============================================================
# PAGE: HELP / DOCUMENTATION
# ============================================================

def page_help():
    st.header("❓ GymIQ Help & Documentation")

    st.markdown(
        """
        ## What GymIQ Does
        GymIQ analyzes:
        - Retail inventory  
        - PT packages  
        - Class packs  
        - Day passes  
        - Lead times  
        - Forecasted demand  
        - Safety stock  
        - Revenue contribution  

        ## Who This Is For
        - Independent gyms  
        - CrossFit boxes  
        - Group fitness studios  
        - University gyms  
        - Personal training facilities  

        ## How to Use GymIQ
        1. Upload your CSV or Excel file  
        2. Go to **Overview** to see your gym health  
        3. Use **Fast/Slow Movers** to remove dead inventory  
        4. Use **Purchase Orders** to restock  
        5. Explore individual items with **SKU Explorer**  
        6. Adjust settings if needed  
        """
    )


# ============================================================
# MAIN APP RUNNER
# ============================================================

def main():
    st.title("🏋️ GymIQ – Inventory & Revenue Intelligence for Gyms")

    # ---- Access code gate ----
    code = st.text_input("Enter access code:", type="password")
    if code != ACCESS_CODE:
        st.stop()

    # ---- Ensure data exists ----
    if not DATA_PATH.exists():
        st.warning("Please upload your data file to begin.")
        page_data_upload()
        return

    # ---- Load data ----
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # ---- Compute metrics ----
    metrics = compute_inventory_metrics(
        df,
        st.session_state["history_days"],
        st.session_state["forecast_days"],
        st.session_state["safety_factor"],
        st.session_state["target_service_days"],
    )

    # ---- Sidebar navigation ----
    menu = st.sidebar.radio(
        "Navigation",
        [
            "Upload Data",
            "Overview",
            "Fast & Slow Movers",
            "Purchase Orders",
            "SKU Explorer",
            "Seasonality",
            "Settings",
            "Raw Data",
            "Help",
        ],
    )

    if menu == "Upload Data":
        page_data_upload()
    elif menu == "Overview":
        page_overview(df, metrics)
    elif menu == "Fast & Slow Movers":
        page_fast_slow(metrics)
    elif menu == "Purchase Orders":
        page_purchase_orders(metrics)
    elif menu == "SKU Explorer":
        page_sku_explorer(df, metrics)
    elif menu == "Seasonality":
        page_seasonality(df)
    elif menu == "Settings":
        page_settings()
    elif menu == "Raw Data":
        page_raw_data(df)
    elif menu == "Help":
        page_help()


if __name__ == "__main__":
    main()


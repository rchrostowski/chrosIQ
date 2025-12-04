import os
import random
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="BarIQ – Inventory Intelligence for Bars & Restaurants",
    layout="wide",
)

# ---- Access code gate ----
ACCESS_CODE = "BAR-TEST"  # change this for real clients

# ---- Data file path (auto-generated if missing) ----
DATA_PATH = Path("data/barIQ_sample.csv")

# ---- Column mappings (same schema as AlcIQ/GymIQ) ----
DATE_COL = "date"                  # transaction date
SKU_COL = "sku"                    # unique SKU code
NAME_COL = "product_name"          # human-readable name
CAT_COL = "category"               # e.g., Draft Beer, Cocktails, Food
SUPPLIER_COL = "supplier"          # distributor / vendor
UNITS_COL = "units_sold"           # units sold on that date
REV_COL = "revenue"                # sales revenue on that date
INV_COL = "inventory_on_hand"      # units on hand end-of-day (bottles, kegs, units)
COST_COL = "unit_cost"             # unit cost
LEADTIME_COL = "lead_time_days"    # lead time in days

# ---- Global defaults (can be overridden in Settings page) ----
DEFAULT_HISTORY_DAYS = 90
DEFAULT_FORECAST_DAYS = 14
DEFAULT_SAFETY_FACTOR = 0.5
DEFAULT_TARGET_SERVICE_DAYS = 21   # how many days of stock we aim to cover
DEFAULT_MIN_SLOW_DAILY_UNITS = 0.02
DEFAULT_FAST_TOP_N = 15
DEFAULT_SLOW_TOP_N = 15

# Initialize Streamlit session state for settings
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
# SAMPLE DATA GENERATOR (BAR / RESTAURANT)
# ============================================================

def generate_sample_dataset(
    path: Path,
    n_skus: int = 260,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
) -> None:
    """
    Generate a realistic bar/restaurant dataset:
    - Draft & packaged beer
    - Wine by glass/bottle
    - Cocktails & shots
    - Spirits bottles
    - NA beverages
    - Food (appetizers, mains)
    - Daily sales, revenue, inventory, cost, lead times
    """
    rng = np.random.default_rng(42)
    random.seed(42)

    dates = pd.date_range(start_date, end_date, freq="D")

    categories = [
        "Draft Beer",
        "Packaged Beer",
        "Wine by Glass",
        "Wine Bottle",
        "Cocktail",
        "Shot",
        "Spirits Bottle",
        "NA Beverage",
        "Food Appetizer",
        "Food Main",
    ]
    cat_weights = np.array([0.18, 0.14, 0.10, 0.06, 0.18, 0.10, 0.06, 0.06, 0.06, 0.06])
    cat_weights = cat_weights / cat_weights.sum()

    suppliers_by_cat: Dict[str, List[str]] = {
        "Draft Beer": ["Local Brewery", "ABInBev", "MillerCoors", "Regional Brewery"],
        "Packaged Beer": ["ABInBev", "Heineken", "Craft Distributor"],
        "Wine by Glass": ["Wine Distributor A", "Wine Distributor B"],
        "Wine Bottle": ["Wine Distributor A", "Wine Importer"],
        "Cocktail": ["House Bar", "Premium Spirits"],
        "Shot": ["House Bar", "Premium Spirits"],
        "Spirits Bottle": ["Spirits Distributor", "Premium Spirits"],
        "NA Beverage": ["CocaCola", "PepsiCo", "NA Vendor"],
        "Food Appetizer": ["Food Supplier A", "Food Supplier B"],
        "Food Main": ["Food Supplier A", "Food Supplier B"],
    }

    base_demand_map = {
        "Draft Beer": (4.0, 15.0),
        "Packaged Beer": (2.0, 8.0),
        "Wine by Glass": (1.0, 5.0),
        "Wine Bottle": (0.1, 0.8),
        "Cocktail": (3.0, 10.0),
        "Shot": (2.0, 8.0),
        "Spirits Bottle": (0.05, 0.4),
        "NA Beverage": (1.0, 4.0),
        "Food Appetizer": (2.0, 10.0),
        "Food Main": (5.0, 20.0),
    }
    cost_map = {
        "Draft Beer": (60, 120),          # keg cost
        "Packaged Beer": (18, 30),       # case cost
        "Wine by Glass": (5, 12),        # per bottle equivalent
        "Wine Bottle": (8, 20),
        "Cocktail": (1.5, 4.0),          # ingredient cost
        "Shot": (0.5, 2.5),
        "Spirits Bottle": (14, 40),
        "NA Beverage": (0.4, 1.0),
        "Food Appetizer": (2.0, 6.0),
        "Food Main": (4.0, 10.0),
    }
    lead_time_map = {
        "Draft Beer": (3, 7),
        "Packaged Beer": (3, 7),
        "Wine by Glass": (5, 10),
        "Wine Bottle": (5, 10),
        "Cocktail": (5, 10),        # assume ingredient ordering
        "Shot": (5, 10),
        "Spirits Bottle": (5, 10),
        "NA Beverage": (3, 7),
        "Food Appetizer": (3, 5),
        "Food Main": (3, 5),
    }

    sku_rows = []
    for i in range(n_skus):
        sku_id = f"SKU-{i+1:04d}"
        cat = rng.choice(categories, p=cat_weights)
        supplier = random.choice(suppliers_by_cat[cat])

        lam_low, lam_high = base_demand_map.get(cat, (0.3, 1.0))
        c_low, c_high = cost_map.get(cat, (5, 20))
        unit_cost = rng.uniform(c_low, c_high)

        # Typical bar markups
        if cat in ["Cocktail", "Shot", "Wine by Glass"]:
            margin_pct = rng.uniform(0.70, 0.82)
        elif cat in ["Draft Beer", "Packaged Beer", "NA Beverage"]:
            margin_pct = rng.uniform(0.55, 0.70)
        elif cat in ["Food Appetizer", "Food Main"]:
            margin_pct = rng.uniform(0.55, 0.65)
        else:
            margin_pct = rng.uniform(0.45, 0.60)

        unit_price = unit_cost * (1 + margin_pct)

        lt_low, lt_high = lead_time_map.get(cat, (5, 10))
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

    name_by_cat = {
        "Draft Beer": ["House Lager Pint", "IPA Pint", "Pale Ale Pint"],
        "Packaged Beer": ["Domestic Bottle", "Import Bottle", "Craft Can"],
        "Wine by Glass": ["House Red Glass", "House White Glass", "Rose Glass"],
        "Wine Bottle": ["Cabernet Bottle", "Pinot Grigio Bottle"],
        "Cocktail": ["Margarita", "Old Fashioned", "Mojito", "Espresso Martini"],
        "Shot": ["Tequila Shot", "Whiskey Shot", "Vodka Shot"],
        "Spirits Bottle": ["Vodka 1L", "Whiskey 750ml", "Tequila 1L"],
        "NA Beverage": ["Soda", "Tonic", "Sparkling Water", "Juice"],
        "Food Appetizer": ["Wings", "Nachos", "Fries", "Tenders"],
        "Food Main": ["Burger", "Chicken Sandwich", "Salad", "Tacos"],
    }

    def make_name(cat: str) -> str:
        return random.choice(name_by_cat.get(cat, ["Bar Item"]))

    sku_df[NAME_COL] = sku_df[CAT_COL].apply(make_name)

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

        # inventory in "units" (glasses equivalent, bottles, plates)
        inventory = int(rng.integers(20, 200))
        reorder_point = int(inventory * 0.35)
        reorder_qty = int(inventory * 0.80)

        for d in dates:
            dow = d.weekday()  # 0=Mon, 6=Sun
            month = d.month

            demand_factor = 1.0

            # Seasonal boost for beer & cocktails in summer
            if month in [6, 7, 8]:
                if cat in ["Draft Beer", "Packaged Beer", "Cocktail", "NA Beverage"]:
                    demand_factor *= 1.4

            # Holiday-ish bump (Dec)
            if month == 12:
                if cat in ["Cocktail", "Shot", "Wine by Glass", "Spirits Bottle"]:
                    demand_factor *= 1.5

            # Weekend pattern
            if dow in [4, 5]:   # Fri, Sat
                demand_factor *= 2.0
            elif dow == 3:      # Thu
                demand_factor *= 1.4
            elif dow == 6:      # Sun
                demand_factor *= 1.2

            lam = max(base_lambda * demand_factor, 0.05)
            units_sold = int(rng.poisson(lam))

            units_sold = min(units_sold, inventory)
            inventory -= units_sold

            # reorder if low
            if inventory < reorder_point:
                inventory += reorder_qty + rng.integers(10, 60)

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


# ============================================================
# DATA LOADING & BASIC UTILITIES
# ============================================================

@st.cache_data(show_spinner="Loading sales & pour data…")
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path.resolve()}")

    df = pd.read_csv(path)

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
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    numeric_cols = [UNITS_COL, REV_COL, INV_COL, COST_COL, LEADTIME_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def get_file_last_updated(path: Path) -> str:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return mtime.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown"


def get_date_range(df: pd.DataFrame) -> Tuple[datetime, datetime]:
    return df[DATE_COL].min(), df[DATE_COL].max()


def add_sidebar_settings():
    st.sidebar.markdown("## Global Settings")
    st.sidebar.caption(
        "These controls change how BarIQ measures demand, risk, and recommended orders. "
        "You can keep the defaults to start."
    )

    st.session_state["history_days"] = st.sidebar.slider(
        "History window (days)",
        min_value=30,
        max_value=365,
        value=st.session_state["history_days"],
        step=15,
        help="How many past days of sales we should use to estimate average daily demand.",
    )
    st.session_state["forecast_days"] = st.sidebar.slider(
        "Forecast horizon (days)",
        min_value=7,
        max_value=60,
        value=st.session_state["forecast_days"],
        step=7,
        help="How far ahead (in days) we want to make sure we have inventory for.",
    )
    st.session_state["safety_factor"] = st.sidebar.slider(
        "Safety stock factor (0.0–1.5)",
        min_value=0.0,
        max_value=1.5,
        value=float(st.session_state["safety_factor"]),
        step=0.1,
        help="Extra buffer on top of expected demand to protect against surprises.",
    )
    st.session_state["target_service_days"] = st.sidebar.slider(
        "Target service coverage (days)",
        min_value=7,
        max_value=60,
        value=st.session_state["target_service_days"],
        step=7,
        help="How many additional days of stock you want on hand after lead time and forecast.",
    )
    st.session_state["min_slow_daily_units"] = st.sidebar.number_input(
        "Min daily units for slow-mover ranking",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["min_slow_daily_units"]),
        step=0.01,
        help="We ignore items that basically never sell when ranking slow movers.",
    )
    st.session_state["fast_top_n"] = st.sidebar.number_input(
        "Top N fast movers",
        min_value=5,
        max_value=50,
        value=int(st.session_state["fast_top_n"]),
        step=5,
        help="How many of your fastest sellers to list.",
    )
    st.session_state["slow_top_n"] = st.sidebar.number_input(
        "Top N slow movers",
        min_value=5,
        max_value=50,
        value=int(st.session_state["slow_top_n"]),
        step=5,
        help="How many of your slowest sellers to list.",
    )


def make_template_df() -> pd.DataFrame:
    """
    Simple template with the required columns and no rows.
    """
    cols = [
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
    return pd.DataFrame(columns=cols)


# ============================================================
# CORE METRICS & ANALYTICS
# ============================================================

def compute_inventory_metrics(
    df: pd.DataFrame,
    history_days: int,
    forecast_days: int,
    safety_factor: float,
    target_service_days: int,
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    max_date = df[DATE_COL].max()
    history_start = max_date - timedelta(days=history_days)

    recent = df[df[DATE_COL] >= history_start].copy()
    daily_units = recent.groupby(SKU_COL)[UNITS_COL].sum() / max(history_days, 1)

    current_inventory = (
        df.sort_values(DATE_COL)
        .groupby(SKU_COL)[INV_COL]
        .last()
        .fillna(0)
    )

    base_info = (
        df.sort_values(DATE_COL)
        .groupby(SKU_COL)
        .agg(
            {
                NAME_COL: "last",
                CAT_COL: "last",
                SUPPLIER_COL: "last",
                COST_COL: "last",
                LEADTIME_COL: "last",
            }
        )
    )

    metrics = base_info.copy()
    metrics["avg_daily_units"] = daily_units.fillna(0)
    metrics["current_inventory"] = current_inventory.fillna(0)
    metrics["forecast_demand"] = metrics["avg_daily_units"] * forecast_days

    metrics[LEADTIME_COL] = metrics[LEADTIME_COL].replace(0, np.nan).fillna(7)
    metrics["safety_stock"] = (
        metrics["avg_daily_units"] * metrics[LEADTIME_COL] * safety_factor
    )

    metrics["weeks_on_hand"] = np.where(
        metrics["avg_daily_units"] > 0,
        metrics["current_inventory"] / (metrics["avg_daily_units"] * 7),
        np.inf,
    )

    metrics["inventory_value"] = metrics["current_inventory"] * metrics[COST_COL]
    metrics["dead_inventory_value"] = np.where(
        metrics["avg_daily_units"] < 0.01, metrics["inventory_value"], 0
    )

    metrics["projected_balance"] = (
        metrics["current_inventory"]
        - metrics["forecast_demand"]
        + metrics["safety_stock"]
    )

    status_list = []
    for _, row in metrics.iterrows():
        inv = row["current_inventory"]
        demand = row["forecast_demand"]
        pbalance = row["projected_balance"]

        if inv <= 0 or pbalance < 0:
            status_list.append("🔥 Stockout Risk")
        elif inv < demand:
            status_list.append("🟡 Low Inventory")
        elif inv > demand * 2:
            status_list.append("🔵 Overstock")
        else:
            status_list.append("✅ Healthy")

    metrics["status"] = status_list

    target_days = forecast_days + metrics[LEADTIME_COL] + target_service_days
    metrics["target_inventory"] = (
        metrics["avg_daily_units"] * target_days + metrics["safety_stock"]
    )

    metrics["recommended_order_qty"] = np.where(
        metrics["status"].isin(["🔥 Stockout Risk", "🟡 Low Inventory"]),
        np.maximum(metrics["target_inventory"] - metrics["current_inventory"], 0),
        0,
    ).round().astype(int)

    recent_rev = recent.groupby(SKU_COL)[REV_COL].sum()
    total_rev = recent_rev.sum()
    share = recent_rev / total_rev if total_rev > 0 else recent_rev * 0
    sorted_share = share.sort_values(ascending=False)
    cum_share = sorted_share.cumsum()

    abc_map: Dict[str, str] = {}
    for sku, cs in cum_share.items():
        if cs <= 0.8:
            abc_map[sku] = "A"
        elif cs <= 0.95:
            abc_map[sku] = "B"
        else:
            abc_map[sku] = "C"

    metrics["abc_class"] = metrics.index.map(lambda x: abc_map.get(x, "C"))
    metrics.reset_index(inplace=True)
    return metrics


def compute_seasonality(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_category_summary(df: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if df.empty or metrics.empty:
        return pd.DataFrame()

    recent = df.copy()
    max_date = recent[DATE_COL].max()
    hist_start = max_date - timedelta(days=st.session_state["history_days"])
    recent_df = recent[recent[DATE_COL] >= hist_start].copy()

    cat_rev = recent_df.groupby(CAT_COL)[REV_COL].sum()
    cat_units = recent_df.groupby(CAT_COL)[UNITS_COL].sum()
    cat_inv_value = metrics.groupby(CAT_COL)["inventory_value"].sum()
    cat_dead_value = metrics.groupby(CAT_COL)["dead_inventory_value"].sum()

    total_rev = cat_rev.sum()
    total_units = cat_units.sum()
    total_inv_value = cat_inv_value.sum()

    cat_df = pd.DataFrame(
        {
            "category": cat_rev.index,
            "revenue": cat_rev.values,
            "units": cat_units.values,
            "inventory_value": cat_inv_value.values,
            "dead_inventory_value": cat_dead_value.values,
        }
    )

    cat_df["revenue_share"] = np.where(
        total_rev > 0, cat_df["revenue"] / total_rev, 0
    )
    cat_df["units_share"] = np.where(
        total_units > 0, cat_df["units"] / total_units, 0
    )
    cat_df["inventory_share"] = np.where(
        total_inv_value > 0, cat_df["inventory_value"] / total_inv_value, 0
    )

    return cat_df.sort_values("revenue", ascending=False)


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


def build_purchase_order(metrics: pd.DataFrame) -> pd.DataFrame:
    po = metrics[
        (metrics["recommended_order_qty"] > 0)
        & (metrics["status"].isin(["🔥 Stockout Risk", "🟡 Low Inventory"]))
    ].copy()
    if po.empty:
        return po

    po["estimated_cost"] = po["recommended_order_qty"] * po[COST_COL]
    po = po[
        [
            SKU_COL,
            NAME_COL,
            CAT_COL,
            SUPPLIER_COL,
            "status",
            "abc_class",
            "avg_daily_units",
            "current_inventory",
            "forecast_demand",
            "recommended_order_qty",
            COST_COL,
            "estimated_cost",
            LEADTIME_COL,
        ]
    ]
    return po.sort_values([SUPPLIER_COL, "status", "estimated_cost"], ascending=[True, True, False])


def simulate_discount(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_skus: List[str],
    discount_pct: float,
    price_elasticity: float = 1.1,
) -> Tuple[pd.DataFrame, float, float]:
    if not selected_skus:
        return pd.DataFrame(), 0.0, 0.0

    df_sim = df[df[SKU_COL].isin(selected_skus)].copy()
    if df_sim.empty:
        return pd.DataFrame(), 0.0, 0.0

    df_sim["price"] = np.where(
        df_sim[UNITS_COL] > 0,
        df_sim[REV_COL] / df_sim[UNITS_COL],
        0,
    )

    base_revenue = df_sim[REV_COL].sum()

    df_sim["new_price"] = df_sim["price"] * (1 - discount_pct)
    df_sim["new_units"] = df_sim[UNITS_COL] * (1 + price_elasticity * discount_pct)
    df_sim["new_revenue"] = df_sim["new_units"] * df_sim["new_price"]

    sku_cost = metrics.set_index(SKU_COL)[COST_COL].to_dict()
    df_sim["unit_cost"] = df_sim[SKU_COL].map(sku_cost).fillna(0)
    df_sim["base_margin"] = (df_sim["price"] - df_sim["unit_cost"]) * df_sim[UNITS_COL]
    df_sim["new_margin"] = (df_sim["new_price"] - df_sim["unit_cost"]) * df_sim["new_units"]

    total_new_revenue = df_sim["new_revenue"].sum()
    delta_revenue = total_new_revenue - base_revenue

    base_margin_total = df_sim["base_margin"].sum()
    new_margin_total = df_sim["new_margin"].sum()
    delta_margin = new_margin_total - base_margin_total

    result = (
        df_sim.groupby(SKU_COL)
        .agg(
            base_revenue=(REV_COL, "sum"),
            new_revenue=("new_revenue", "sum"),
            base_units=(UNITS_COL, "sum"),
            new_units=("new_units", "sum"),
            base_margin=("base_margin", "sum"),
            new_margin=("new_margin", "sum"),
        )
        .reset_index()
    )

    return result, float(delta_revenue), float(delta_margin)


# ============================================================
# KPI + UI HELPERS
# ============================================================

def kpi_header(df: pd.DataFrame, metrics: pd.DataFrame):
    total_rev = df[REV_COL].sum()
    total_units = df[UNITS_COL].sum()
    unique_skus = df[SKU_COL].nunique()
    total_inv_value = metrics["inventory_value"].sum()
    dead_value = metrics["dead_inventory_value"].sum()
    stockout_count = (metrics["status"] == "🔥 Stockout Risk").sum()
    overstock_count = (metrics["status"] == "🔵 Overstock").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_rev:,.0f}")
    c2.metric("Total Units Poured/Sold", f"{total_units:,.0f}")
    c3.metric("Active SKUs/Menu Items", f"{unique_skus:,}")
    c4.metric("On-Hand Inventory Value", f"${total_inv_value:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Dead Inventory Value", f"${dead_value:,.0f}")
    c6.metric("🔥 Stockout Risks", f"{stockout_count}")
    c7.metric("🔵 Overstock Items", f"{overstock_count}")


# ============================================================
# PAGE FUNCTIONS
# ============================================================

def page_overview(df: pd.DataFrame, metrics: pd.DataFrame):
    st.subheader("Overview")
    st.caption(
        "High-level health of your bar: revenue, inventory value, and how many items are at risk "
        "of running out or sitting too long on the shelf."
    )
    kpi_header(df, metrics)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Revenue Trend by Month & Category")
        st.caption(
            "See how bar sales change over time by category. Look for trends (summer patio spike, holidays) "
            "and which categories are driving the most dollars."
        )
        df_month = (
            df.set_index(DATE_COL)
            .groupby([pd.Grouper(freq="M"), CAT_COL])[REV_COL]
            .sum()
            .reset_index()
        )
        if df_month.empty:
            st.info("No revenue data available.")
        else:
            df_month["month"] = df_month[DATE_COL].dt.to_period("M").astype(str)
            pivot = df_month.pivot(index="month", columns=CAT_COL, values=REV_COL).fillna(0)
            st.area_chart(pivot)

    with col2:
        st.markdown("#### Inventory Status Breakdown")
        st.caption(
            "How many SKUs are healthy, low, at risk of stockout, or overstocked. "
            "Red and yellow are where you should focus first."
        )
        status_counts = metrics["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        if status_counts.empty:
            st.info("No inventory metrics available.")
        else:
            st.bar_chart(status_counts.set_index("status"))

        st.markdown("#### ABC Revenue Classes")
        st.caption(
            "A, B, C shows which SKUs drive most of your revenue. "
            "A-items (top ~80% of sales) deserve the most attention."
        )
        abc_counts = metrics["abc_class"].value_counts().reindex(["A", "B", "C"]).fillna(0)
        st.bar_chart(abc_counts)


def page_inventory_forecast(metrics: pd.DataFrame):
    st.subheader("Inventory Forecast & Risk")
    st.caption(
        "This view turns recent sales into a forward-looking forecast. "
        "Use it to see which beers, wines, cocktails, and food items are at risk of running out or are overstocked."
    )

    status_filter = st.multiselect(

# ------------------------------------------------------------------------------
# Footer – Privacy
# ------------------------------------------------------------------------------

st.divider()
st.markdown(
    """
### 🔐 Privacy & Data

Your data is **100% private**.

barIQ does **not** store, save, log, transmit, share, or sell any of your data — ever.

Your Excel file (and any online orders CSV) are processed only while this app is open.
Once you close or refresh, the data is gone.

We cannot see your sales numbers.  
We cannot see your inventory.  
We cannot see your product file.  

Your information belongs entirely to **you** — barIQ simply analyzes it to generate smarter order recommendations.
"""
)
st.caption("barIQ – Inventory intelligence for modern bars and restaurants.")


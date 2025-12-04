import random
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# BASIC CONFIG
# ============================================================

st.set_page_config(
    page_title="BarIQ – Inventory Intelligence for Bars & Restaurants",
    layout="wide",
)

# Access code
ACCESS_CODE = "BAR-TEST"

# Data path
DATA_PATH = Path("data/barIQ_sample.csv")

# Column names (shared schema)
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

# Default model settings
DEFAULT_HISTORY_DAYS = 90
DEFAULT_FORECAST_DAYS = 14
DEFAULT_SAFETY_FACTOR = 0.5
DEFAULT_TARGET_SERVICE_DAYS = 21
DEFAULT_MIN_SLOW_DAILY_UNITS = 0.02
DEFAULT_FAST_TOP_N = 15
DEFAULT_SLOW_TOP_N = 15

for key, val in {
    "history_days": DEFAULT_HISTORY_DAYS,
    "forecast_days": DEFAULT_FORECAST_DAYS,
    "safety_factor": DEFAULT_SAFETY_FACTOR,
    "target_service_days": DEFAULT_TARGET_SERVICE_DAYS,
    "min_slow_daily_units": DEFAULT_MIN_SLOW_DAILY_UNITS,
    "fast_top_n": DEFAULT_FAST_TOP_N,
    "slow_top_n": DEFAULT_SLOW_TOP_N,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================
# SAMPLE DATA GENERATOR (BAR / RESTAURANT)
# ============================================================

def generate_sample_dataset(
    path: Path,
    n_skus: int = 240,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
) -> None:
    """
    Generates a realistic bar/restaurant dataset:
    - Draft & packaged beer, wine, cocktails, shots, spirits, NA, food.
    - Weekend + seasonal patterns.
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
    cat_weights /= cat_weights.sum()

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
        "Draft Beer": (60, 120),       # keg
        "Packaged Beer": (18, 30),     # case
        "Wine by Glass": (5, 12),
        "Wine Bottle": (8, 20),
        "Cocktail": (1.5, 4.0),        # ingredient cost
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
        "Cocktail": (5, 10),
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

        # Bar-style margins
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
        "Wine by Glass": ["House Red Glass", "House White Glass", "Rosé Glass"],
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

        inventory = int(rng.integers(20, 200))
        reorder_point = int(inventory * 0.35)
        reorder_qty = int(inventory * 0.8)

        for d in dates:
            dow = d.weekday()   # 0=Mon,6=Sun
            month = d.month

            demand_factor = 1.0

            # Summer patio / beer & cocktail bump
            if month in [6, 7, 8]:
                if cat in ["Draft Beer", "Packaged Beer", "Cocktail", "NA Beverage"]:
                    demand_factor *= 1.4

            # Holiday bump in December
            if month == 12 and cat in ["Cocktail", "Shot", "Wine by Glass", "Spirits Bottle"]:
                demand_factor *= 1.5

            # Weekend pattern
            if dow in [4, 5]:  # Fri / Sat
                demand_factor *= 2.0
            elif dow == 3:    # Thu
                demand_factor *= 1.4
            elif dow == 6:    # Sun
                demand_factor *= 1.2

            lam = max(base_lambda * demand_factor, 0.05)
            units_sold = int(rng.poisson(lam))

            units_sold = min(units_sold, inventory)
            inventory -= units_sold

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
# DATA LOADING & UTILITIES
# ============================================================

@st.cache_data(show_spinner="Loading bar data…")
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
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    for col in [UNITS_COL, REV_COL, INV_COL, COST_COL, LEADTIME_COL]:
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


def make_template_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


# ============================================================
# CORE METRICS
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
    hist_start = max_date - timedelta(days=history_days)
    recent = df[df[DATE_COL] >= hist_start].copy()

    daily_units = recent.groupby(SKU_COL)[UNITS_COL].sum() / max(history_days, 1)
    current_inv = (
        df.sort_values(DATE_COL)
        .groupby(SKU_COL)[INV_COL]
        .last()
        .fillna(0)
    )

    base = (
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

    metrics = base.copy()
    metrics["avg_daily_units"] = daily_units.fillna(0)
    metrics["current_inventory"] = current_inv.fillna(0)
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
        metrics["avg_daily_units"] < 0.01,
        metrics["inventory_value"],
        0,
    )

    metrics["projected_balance"] = (
        metrics["current_inventory"]
        - metrics["forecast_demand"]
        + metrics["safety_stock"]
    )

    status_list = []
    for _, r in metrics.iterrows():
        inv = r["current_inventory"]
        demand = r["forecast_demand"]
        pb = r["projected_balance"]

        if inv <= 0 or pb < 0:
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

    # ABC classes by revenue
    recent_rev = recent.groupby(SKU_COL)[REV_COL].sum()
    total_rev = recent_rev.sum()
    share = recent_rev / total_rev if total_rev > 0 else recent_rev * 0
    cum_share = share.sort_values(ascending=False).cumsum()

    abc_map: Dict[str, str] = {}
    for sku, cs in cum_share.items():
        if cs <= 0.8:
            abc_map[sku] = "A"
        elif cs <= 0.95:
            abc_map[sku] = "B"
        else:
            abc_map[sku] = "C"

    metrics["abc_class"] = metrics.index.map(lambda s: abc_map.get(s, "C"))
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
    max_date = df[DATE_COL].max()
    hist_start = max_date - timedelta(days=st.session_state["history_days"])
    recent = df[df[DATE_COL] >= hist_start].copy()

    cat_rev = recent.groupby(CAT_COL)[REV_COL].sum()
    cat_units = recent.groupby(CAT_COL)[UNITS_COL].sum()
    cat_inv_val = metrics.groupby(CAT_COL)["inventory_value"].sum()
    cat_dead_val = metrics.groupby(CAT_COL)["dead_inventory_value"].sum()

    total_rev = cat_rev.sum()
    total_units = cat_units.sum()
    total_inv = cat_inv_val.sum()

    out = pd.DataFrame(
        {
            CAT_COL: cat_rev.index,
            "revenue": cat_rev.values,
            "units": cat_units.values,
            "inventory_value": cat_inv_val.values,
            "dead_inventory_value": cat_dead_val.values,
        }
    )
    out["revenue_share"] = np.where(total_rev > 0, out["revenue"] / total_rev, 0)
    out["units_share"] = np.where(total_units > 0, out["units"] / total_units, 0)
    out["inventory_share"] = np.where(total_inv > 0, out["inventory_value"] / total_inv, 0)
    return out.sort_values("revenue", ascending=False)


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
        .sort_values("avg_daily_units")
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
    cols = [
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
    return po[cols].sort_values([SUPPLIER_COL, "status", "estimated_cost"], ascending=[True, True, False])


# ============================================================
# KPI HEADER
# ============================================================

def kpi_header(df: pd.DataFrame, metrics: pd.DataFrame):
    total_rev = df[REV_COL].sum()
    total_units = df[UNITS_COL].sum()
    skus = df[SKU_COL].nunique()
    inv_val = metrics["inventory_value"].sum()
    dead_val = metrics["dead_inventory_value"].sum()
    stockouts = (metrics["status"] == "🔥 Stockout Risk").sum()
    overstock = (metrics["status"] == "🔵 Overstock").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_rev:,.0f}")
    c2.metric("Total Units Sold/Poured", f"{total_units:,.0f}")
    c3.metric("Active SKUs", f"{skus:,}")
    c4.metric("Inventory Value", f"${inv_val:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Dead Inventory Value", f"${dead_val:,.0f}")
    c6.metric("🔥 Stockout Risks", f"{stockouts}")
    c7.metric("🔵 Overstock SKUs", f"{overstock}")


# ============================================================
# PAGES
# ============================================================

def page_data_upload():
    st.header("📁 Your Data (Upload & Template)")
    st.markdown(
        """
        Start here. BarIQ needs **daily sales and inventory** from your POS/back office.

        1. Upload your CSV/Excel export below.  
        2. Or download the template and map your data into it.
        """
    )

    uploaded = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                df_new = pd.read_excel(uploaded)
            else:
                df_new = pd.read_csv(uploaded)

            required = make_template_df().columns
            missing = [c for c in required if c not in df_new.columns]
            if missing:
                st.error(f"Uploaded file is missing required columns: {missing}")
            else:
                df_new[DATE_COL] = pd.to_datetime(df_new[DATE_COL], errors="coerce")
                df_new = df_new.dropna(subset=[DATE_COL])
                for col in [UNITS_COL, REV_COL, INV_COL, COST_COL, LEADTIME_COL]:
                    df_new[col] = pd.to_numeric(df_new[col], errors="coerce").fillna(0.0)
                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                df_new.to_csv(DATA_PATH, index=False)
                st.success("Data uploaded and saved. BarIQ is now using your file.")
                st.rerun()
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    st.markdown("---")
    st.subheader("⬇️ Download Template")
    tmpl = make_template_df()
    csv_bytes = tmpl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV Template",
        data=csv_bytes,
        file_name="bariq_template.csv",
        mime="text/csv",
    )

    try:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            tmpl.to_excel(writer, index=False, sheet_name="Template")
        buf.seek(0)
        st.download_button(
            "Download Excel Template (.xlsx)",
            data=buf,
            file_name="bariq_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.info("Excel export not available here. Use the CSV template instead.")


def page_overview(df: pd.DataFrame, metrics: pd.DataFrame):
    st.header("📊 Overview")
    st.markdown(
        """
        This is your **bar at a glance**: total revenue, inventory value, and how many items are
        at risk of running out or just sitting on the shelf too long.
        """
    )
    kpi_header(df, metrics)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Revenue by Month & Category")
        st.caption("See how sales evolve over time by category (draft, cocktails, food, etc.).")
        df_m = (
            df.set_index(DATE_COL)
            .groupby([pd.Grouper(freq="M"), CAT_COL])[REV_COL]
            .sum()
            .reset_index()
        )
        if df_m.empty:
            st.info("No revenue data yet.")
        else:
            df_m["month"] = df_m[DATE_COL].dt.to_period("M").astype(str)
            pivot = df_m.pivot(index="month", columns=CAT_COL, values=REV_COL).fillna(0)
            st.area_chart(pivot)

    with col2:
        st.subheader("Inventory Status Breakdown")
        status_counts = metrics["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        if status_counts.empty:
            st.info("No inventory metrics yet.")
        else:
            st.bar_chart(status_counts.set_index("status"))

        st.subheader("ABC Revenue Classes")
        abc_counts = metrics["abc_class"].value_counts().reindex(["A", "B", "C"]).fillna(0)
        st.bar_chart(abc_counts)


def page_inventory_forecast(metrics: pd.DataFrame):
    st.header("📈 Inventory Forecast & Risk")
    st.markdown(
        """
        This page turns recent sales into a **forward-looking forecast**.

        Use it to answer:
        - What are we about to **86**?  
        - Where do we have **too much** tied up in inventory?  
        """
    )

    status_filter = st.multiselect(
        "Filter by Status",
        options=["🔥 Stockout Risk", "🟡 Low Inventory", "🔵 Overstock", "✅ Healthy"],
        default=["🔥 Stockout Risk", "🟡 Low Inventory"],
    )

    cat_options = sorted(metrics[CAT_COL].dropna().unique())
    cat_filter = st.multiselect(
        "Filter by Category",
        options=cat_options,
        default=[],
    )

    abc_filter = st.multiselect(
        "Filter by ABC Class",
        options=["A", "B", "C"],
        default=["A", "B", "C"],
    )

    view = metrics.copy()
    if status_filter:
        view = view[view["status"].isin(status_filter)]
    if cat_filter:
        view = view[view[CAT_COL].isin(cat_filter)]
    if abc_filter:
        view = view[view["abc_class"].isin(abc_filter)]

    if view.empty:
        st.success("No SKUs match these filters – that might actually be a good thing 😄")
        return

    cols = [
        SKU_COL,
        NAME_COL,
        CAT_COL,
        SUPPLIER_COL,
        "status",
        "abc_class",
        "avg_daily_units",
        "current_inventory",
        "forecast_demand",
        "weeks_on_hand",
        "recommended_order_qty",
        "inventory_value",
    ]
    st.dataframe(view[cols].sort_values(["status", "weeks_on_hand"]), use_container_width=True)

    st.caption(
        "- **Avg daily units**: average units sold per day.\n"
        "- **Weeks on hand**: how long current inventory lasts at that pace.\n"
        "- **Recommended order qty**: what BarIQ suggests ordering to get back to a safe level."
    )


def page_fast_slow(metrics: pd.DataFrame):
    st.header("⚡ Fast Movers & 🐢 Slow Movers")
    st.markdown(
        """
        **Fast movers**: lines and menu items that drive most of your sales.  
        **Slow movers**: items that barely sell but still take up space and cash.
        """
    )

    slow, fast = get_slow_fast_movers(
        metrics,
        slow_n=st.session_state["slow_top_n"],
        fast_n=st.session_state["fast_top_n"],
        min_slow_daily_units=st.session_state["min_slow_daily_units"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐢 Slow Movers")
        if slow.empty:
            st.info("No slow movers for current settings.")
        else:
            st.dataframe(
                slow[
                    [
                        SKU_COL,
                        NAME_COL,
                        CAT_COL,
                        "abc_class",
                        "avg_daily_units",
                        "current_inventory",
                        "weeks_on_hand",
                        "inventory_value",
                    ]
                ],
                use_container_width=True,
            )

    with col2:
        st.subheader("⚡ Fast Movers")
        if fast.empty:
            st.info("No fast movers found (not enough data).")
        else:
            st.dataframe(
                fast[
                    [
                        SKU_COL,
                        NAME_COL,
                        CAT_COL,
                        "abc_class",
                        "avg_daily_units",
                        "current_inventory",
                        "weeks_on_hand",
                        "inventory_value",
                    ]
                ],
                use_container_width=True,
            )


def page_purchase_orders(metrics: pd.DataFrame):
    st.header("📦 Purchase Order Builder")
    st.markdown(
        """
        BarIQ pulls together **everything that’s at risk** and turns it into a ready-to-send
        purchase order you can email to distributors.
        """
    )

    po = build_purchase_order(metrics)
    if po.empty:
        st.success("No current stockout/low inventory risks that need ordering.")
        return

    suppliers = sorted(po[SUPPLIER_COL].dropna().unique())
    supplier_filter = st.multiselect(
        "Filter by Supplier", options=suppliers, default=suppliers
    )
    cat_filter = st.multiselect(
        "Filter by Category",
        options=sorted(po[CAT_COL].dropna().unique()),
        default=[],
    )

    view = po.copy()
    if supplier_filter:
        view = view[view[SUPPLIER_COL].isin(supplier_filter)]
    if cat_filter:
        view = view[view[CAT_COL].isin(cat_filter)]

    if view.empty:
        st.warning("No items match the current filters.")
        return

    st.dataframe(view, use_container_width=True)
    total_cost = view["estimated_cost"].sum()
    st.markdown(f"**Total Estimated PO Cost:** ${total_cost:,.0f}")

    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Purchase Order CSV",
        data=csv_bytes,
        file_name=f"bariq_purchase_order_{datetime.today().date()}.csv",
        mime="text/csv",
    )


def page_sku_explorer(df: pd.DataFrame, metrics: pd.DataFrame):
    st.header("🔍 SKU Explorer")
    st.markdown(
        """
        Drill into any SKU: see its sales trend, revenue, and current risk level.
        Great for “should we 86 this?” or “why is this always out?” conversations.
        """
    )

    sku_list = sorted(metrics[SKU_COL].unique())
    selected = st.selectbox("Select a SKU", options=sku_list)

    row_m = metrics[metrics[SKU_COL] == selected].iloc[0]
    sub = df[df[SKU_COL] == selected].sort_values(DATE_COL)

    st.subheader(f"{row_m[NAME_COL]} ({row_m[SKU_COL]})")
    st.caption(f"Category: {row_m[CAT_COL]} • Supplier: {row_m[SUPPLIER_COL]} • ABC: {row_m['abc_class']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", row_m["status"])
    c2.metric("Avg Daily Units", f"{row_m['avg_daily_units']:.2f}")
    c3.metric(
        "Weeks on Hand",
        "∞" if np.isinf(row_m["weeks_on_hand"]) else f"{row_m['weeks_on_hand']:.1f}",
    )
    c4.metric("Current Inventory", f"{row_m['current_inventory']:,.0f}")

    if sub.empty:
        st.info("No sales history for this SKU.")
        return

    st.subheader("Units Sold per Week")
    weekly_units = sub.set_index(DATE_COL)[UNITS_COL].resample("W").sum()
    st.line_chart(weekly_units)

    st.subheader("Revenue per Week")
    weekly_rev = sub.set_index(DATE_COL)[REV_COL].resample("W").sum()
    st.line_chart(weekly_rev)

    st.subheader("Raw Daily Records")
    st.dataframe(
        sub[[DATE_COL, UNITS_COL, REV_COL, INV_COL]],
        use_container_width=True,
    )


def page_category_analytics(df: pd.DataFrame, metrics: pd.DataFrame):
    st.header("📂 Category & Supplier Analytics")
    st.markdown(
        """
        Which categories and suppliers actually move the needle for your bar?  
        Use this to prioritize what you manage closely and who you negotiate with.
        """
    )

    cat_summary = compute_category_summary(df, metrics)
    if cat_summary.empty:
        st.info("Not enough data to build category summary.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category KPIs")
        st.dataframe(
            cat_summary[
                [
                    CAT_COL,
                    "revenue",
                    "revenue_share",
                    "units",
                    "units_share",
                    "inventory_value",
                    "inventory_share",
                    "dead_inventory_value",
                ]
            ],
            use_container_width=True,
        )

    with col2:
        st.subheader("Revenue Share by Category")
        st.bar_chart(cat_summary.set_index(CAT_COL)["revenue_share"])
        st.subheader("Inventory Value by Category")
        st.bar_chart(cat_summary.set_index(CAT_COL)["inventory_value"])

    st.markdown("---")
    st.subheader("Supplier Exposure")
    supplier_inv = (
        metrics.groupby(SUPPLIER_COL)["inventory_value"]
        .sum()
        .sort_values(ascending=False)
    )
    supplier_rev = (
        df.groupby(SUPPLIER_COL)[REV_COL]
        .sum()
        .sort_values(ascending=False)
    )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("Inventory Value by Supplier")
        st.bar_chart(supplier_inv)
    with c4:
        st.markdown("Revenue by Supplier")
        st.bar_chart(supplier_rev)


def page_seasonality(df: pd.DataFrame, metrics: pd.DataFrame):
    st.header("📆 Seasonality & Trends")
    st.markdown(
        """
        Some items are **summer patio heroes**, others are **winter warmers**.  
        Seasonality tells you which SKUs spike at different times of year so you can plan ahead.
        """
    )

    sea = compute_seasonality(df)
    if sea.empty:
        st.info("Not enough data to compute seasonality.")
        return

    merged = pd.merge(
        metrics[[SKU_COL, NAME_COL, CAT_COL]],
        sea,
        on=SKU_COL,
        how="left",
    )

    top_n = st.slider("Show top N most seasonal SKUs", 10, 100, 25, step=5)
    top_seasonal = merged.sort_values("seasonality_score", ascending=False).head(top_n)
    st.subheader("Most Seasonal SKUs")
    st.dataframe(
        top_seasonal[[SKU_COL, NAME_COL, CAT_COL, "seasonality_score"]],
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Seasonality Example: Pick a SKU")
    sku_options = top_seasonal[SKU_COL].tolist()
    if not sku_options:
        st.info("No seasonal SKUs detected.")
        return

    selected = st.selectbox("Pick a seasonal SKU", options=sku_options)
    sub = df[df[SKU_COL] == selected].copy()
    if sub.empty:
        st.info("No data for this SKU.")
        return

    sub["month"] = sub[DATE_COL].dt.month
    month_units = sub.groupby("month")[UNITS_COL].sum()
    st.line_chart(month_units)


def page_settings(df: pd.DataFrame, metrics: pd.DataFrame):
    st.header("⚙️ Settings & Assumptions")
    st.markdown(
        """
        Tune how conservative or aggressive BarIQ should be.  
        You can always hit **Reset** to go back to defaults.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Demand & Risk")
        st.session_state["history_days"] = st.number_input(
            "History window (days)",
            min_value=30,
            max_value=365,
            value=int(st.session_state["history_days"]),
            step=15,
        )
        st.session_state["forecast_days"] = st.number_input(
            "Forecast horizon (days)",
            min_value=7,
            max_value=60,
            value=int(st.session_state["forecast_days"]),
            step=7,
        )
        st.session_state["safety_factor"] = st.slider(
            "Safety stock factor",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state["safety_factor"]),
            step=0.1,
        )

    with c2:
        st.subheader("Service & Slow/Fast Definitions")
        st.session_state["target_service_days"] = st.number_input(
            "Target service coverage (days)",
            min_value=7,
            max_value=60,
            value=int(st.session_state["target_service_days"]),
            step=7,
        )
        st.session_state["min_slow_daily_units"] = st.number_input(
            "Min daily units for slow movers",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["min_slow_daily_units"]),
            step=0.01,
        )
        st.session_state["slow_top_n"] = st.number_input(
            "Slow movers shown",
            min_value=5,
            max_value=50,
            value=int(st.session_state["slow_top_n"]),
            step=5,
        )
        st.session_state["fast_top_n"] = st.number_input(
            "Fast movers shown",
            min_value=5,
            max_value=50,
            value=int(st.session_state["fast_top_n"]),
            step=5,
        )

    st.markdown("---")
    st.subheader("Data Info")
    start, end = get_date_range(df)
    st.write(f"Date range: **{start.date()} → {end.date()}**")
    st.write(f"Rows: **{len(df):,}** • SKUs: **{df[SKU_COL].nunique():,}**")

    if st.button("Reset Settings to Defaults"):
        st.session_state["history_days"] = DEFAULT_HISTORY_DAYS
        st.session_state["forecast_days"] = DEFAULT_FORECAST_DAYS
        st.session_state["safety_factor"] = DEFAULT_SAFETY_FACTOR
        st.session_state["target_service_days"] = DEFAULT_TARGET_SERVICE_DAYS
        st.session_state["min_slow_daily_units"] = DEFAULT_MIN_SLOW_DAILY_UNITS
        st.session_state["fast_top_n"] = DEFAULT_FAST_TOP_N
        st.session_state["slow_top_n"] = DEFAULT_SLOW_TOP_N
        st.success("Settings reset. Refreshing…")
        st.cache_data.clear()
        st.rerun()


def page_raw_data(df: pd.DataFrame):
    st.header("📄 Raw Data")
    st.markdown(
        "For power users. This is the exact data BarIQ uses under the hood."
    )
    st.dataframe(df, use_container_width=True)


def page_help():
    st.header("❓ Help & FAQ")
    st.markdown(
        f"""
### What does BarIQ do?

BarIQ is an **inventory & margin intelligence tool for bars and restaurants**.  
It helps you:
- Catch **stockout risks** before busy nights  
- Spot **slow movers** and **dead inventory** tying up cash  
- Identify **fast movers** that deserve prime tap lines and menu space  
- Build **purchase orders** in minutes instead of hours  
- Understand **seasonality** in drinks and food  

### What data do I need?

One row per SKU per day, with these columns:

- `{DATE_COL}` – calendar date  
- `{SKU_COL}` – your internal code or POS ID  
- `{NAME_COL}` – what guests / staff call it  
- `{CAT_COL}` – e.g., Draft Beer, Cocktail, Food Main  
- `{SUPPLIER_COL}` – which distributor/wholesaler it comes from  
- `{UNITS_COL}` – units sold/poured that day  
- `{REV_COL}` – revenue from that SKU that day  
- `{INV_COL}` – inventory on hand (units)  
- `{COST_COL}` – cost per unit  
- `{LEADTIME_COL}` – days between order and delivery  

You can always download the template on the **Your Data** page.
        """
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("BarIQ – Inventory Intelligence for Bars & Restaurants")
    st.caption(
        "Plug in your sales & inventory data and BarIQ will tell you what's about to 86, "
        "what's overstocked, and what to order next."
    )

    # Access gate
    code = st.text_input("Enter access code:", type="password")
    if code != ACCESS_CODE:
        st.stop()

    # Ensure data exists
    if not DATA_PATH.exists():
        st.info("No data file found. Creating a realistic sample bar dataset for demo purposes.")
        generate_sample_dataset(DATA_PATH)

    # Top bar
    top_c1, top_c2 = st.columns([1, 3])
    with top_c1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with top_c2:
        st.caption(f"Data source: `{DATA_PATH}` • Last updated: {get_file_last_updated(DATA_PATH)}")

    # Load data
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar nav
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "",
        [
            "Your Data (Upload & Template)",
            "Overview",
            "Inventory Forecast",
            "Fast & Slow Movers",
            "Purchase Orders",
            "SKU Explorer",
            "Category & Supplier Analytics",
            "Seasonality",
            "Settings",
            "Raw Data",
            "Help",
        ],
    )

    # Compute metrics
    metrics = compute_inventory_metrics(
        df,
        history_days=int(st.session_state["history_days"]),
        forecast_days=int(st.session_state["forecast_days"]),
        safety_factor=float(st.session_state["safety_factor"]),
        target_service_days=int(st.session_state["target_service_days"]),
    )

    # Route
    if page == "Your Data (Upload & Template)":
        page_data_upload()
    elif page == "Overview":
        page_overview(df, metrics)
    elif page == "Inventory Forecast":
        page_inventory_forecast(metrics)
    elif page == "Fast & Slow Movers":
        page_fast_slow(metrics)
    elif page == "Purchase Orders":
        page_purchase_orders(metrics)
    elif page == "SKU Explorer":
        page_sku_explorer(df, metrics)
    elif page == "Category & Supplier Analytics":
        page_category_analytics(df, metrics)
    elif page == "Seasonality":
        page_seasonality(df, metrics)
    elif page == "Settings":
        page_settings(df, metrics)
    elif page == "Raw Data":
        page_raw_data(df)
    elif page == "Help":
        page_help()


if __name__ == "__main__":
    main()


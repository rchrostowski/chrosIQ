# 💪 gymIQ – Gym Retail & Inventory Optimizer

gymIQ helps gyms and fitness centers build **clean, data-driven vendor orders** for their pro shop and front desk inventory.

Instead of manually guessing how much protein, energy drinks, bars, apparel, and accessories to order, gymIQ uses your **real sales + inventory data** to recommend:

- Optimal order quantities by SKU
- Stockout-risk alerts (fast movers with low stock)
- Slow-mover / overstock warnings
- Clean vendor-ready order files (CSV + Excel)

---

## 🚀 How It Works

1. **Download the Excel template** from the app sidebar  
2. Export or copy data from your POS into the template:
   - `Sales` – daily sales per product
   - `Inventory` – current on-hand stock
   - `Products` – master list of products (brand, category, vendor, cost, etc.)
3. **Upload the completed Excel file** into gymIQ  
4. Review:
   - Recommended Order
   - Inventory Health
   - Product Explorer
   - Vendor Summary
   - Clean Order Report
5. Download a **vendor-ready order file** and send it to your suppliers.

---

## 📂 Required Excel Structure

### `Sales` sheet

Columns:

- `date` – Date of sale (YYYY-MM-DD)  
- `sku` – Product ID / code used in your system  
- `product_name` – Name of the item  
- `qty_sold` – Units sold on that day  
- `unit_price` – Price charged to member

### `Inventory` sheet

Columns:

- `sku` – Product ID  
- `on_hand_qty` – Units currently in stock  

### `Products` sheet

Columns:

- `sku` – Product ID (must match Sales & Inventory)  
- `brand` – Brand (e.g., Optimum Nutrition, Alani Nu, GymShark)  
- `product_name` – Full product name  
- `category` – Supplement, Drink, Bar, Apparel, Accessory, etc.  
- `size` – 2lb, 16oz can, box of 12, S/M/L, etc.  
- `vendor` – Supplier / distributor  
- `cost` – Your cost per unit  
- `case_size` – Units per case (if you order by case)  
- `lead_time_days` – Typical delivery lead time from that vendor  

The app includes a **Download Template** button so gyms can start from a clean file.

---

## 🧮 Logic (High Level)

- gymIQ calculates **average daily demand** per product using:
  - A longer historical window (e.g., last 30 days)
  - A shorter recent window (e.g., last 7 days)
  - A blended average that reacts to recent changes
- It then computes:
  - **Reorder point** based on lead time + demand + volatility  
  - **Target stock** based on lead time + order frequency  
  - **Recommended order quantity** (rounded to cases if needed)

---

## 🔐 Privacy

gymIQ does **not** store, save, or sell any data.  
Your Excel file is processed only in your session.

---

## ▶️ Running Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py

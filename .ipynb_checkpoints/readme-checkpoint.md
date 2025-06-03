# 📦 Multi-Location Sales Forecasting & Inventory Optimization

This project provides a **robust and extensible workflow** for sales forecasting, automated replenishment, and inventory health analysis for multi-location retailers—built for both data science exploration and real-world operations.

---

## 🚀 **Project Overview**

- **Forecast sales and automate inventory decisions** per item, per location, using machine learning (Prophet) and best-practice statistical rules.
- **Incorporate real business constraints:** account for demand volatility, service level, returns, and more.
- **Simulate “what if” scenarios**—see how changing service level affects inventory risk and cashflow.

---

## 🏗️ **Pipeline & Features**

### 1. **Data Preparation**
- **Load transaction data**: sales, returns, variations, item info.
- **Clean and filter**:
    - Exclude returns (negative sales).
    - Filter only items with valid variation IDs.
    - Restrict to the most recent 12 months for forecasting relevance.

### 2. **Daily Aggregation**
- Aggregate sales per date, location, and variation.
- Remove or track negative sales (returns) for reporting and quality control.

### 3. **Feature Engineering & Quality Control**
- Count **number of sales days and sales weeks** per item/location.
- Attach item/variation names and business metadata.
- **Set `enough_history` flag:** forecast only if item has, e.g., ≥20 sales days in ≥4 different weeks (prevents forecasting on “bursty” or sparse data).

### 4. **Demand Volatility and Adaptive Safety Stock**
- Calculate **coefficient of variation (CV)** for each SKU-location.
- Assign **z-scores** (service level multipliers) dynamically:
    - Higher z-score (buffer) for volatile or high-value items.
    - Lower z-score for steady sellers.
- **Classic safety stock formula:**  
    `Safety Stock = z_score × σ`  
    (where σ is estimated from Prophet’s prediction intervals or historical demand).

### 5. **Forecasting Logic**
- **For each item/location pair:**
    - If `enough_history` is `True`: use [Facebook Prophet](https://facebook.github.io/prophet/) for sales forecasting.
    - If `enough_history` is `False`: fallback to recent average or simple rules.
- **Forecast output:**
    - `reorder_level` (when to order)
    - `replenish_level` (how much to bring in)
    - All relevant metadata for reporting and integration

### 6. **Simulation and Scenario Analysis**
- **Simulate the effect of different service levels** (`z_score`): 90% (1.28), 95% (1.65), 99% (2.33) service.
- Compare stock buffers and risk of stockout as you adjust your policy.
- Visualize actual vs. forecasted sales for random SKUs.

### 7. **Obsolescence and Inventory Health**
- Track **days since last sale** per item/location.
- Flag items as **potentially obsolete** if not sold in the last 60 days.

---

## 🧮 **Key Formulas**

- **Safety Stock:**  
  `Safety Stock = z_score × σ`  
  *(σ is the standard deviation of demand over lead time, estimated from Prophet or history.)*
- **Reorder Point:**  
  `Reorder Level = Lead Time Demand + Safety Stock`
- **Replenish Level:**  
  `Replenish Level = Reorder Level + Lead Time Demand`
- **Prophet σ estimation:**  
  `σ = (yhat_upper - yhat_lower) / 3.29`  
  *(for a 99.9% prediction interval; adjust 3.29 for other confidence levels)*

---

## 📊 **How To Use**

1. **Prepare your input CSVs** (`sales`, `sales_items`, `variations`, etc.).
2. **Run the notebook** step by step—cells are modular, with all preprocessing upfront.
3. **Adjust business rules** as needed (min days/weeks, z-scores, obsolescence windows).
4. **Check output tables and charts.**  
    - Export results as CSV for downstream reporting or upload to cloud.
5. **Try simulation cells** at the end to see impact of different service levels on inventory.

---

## 🧠 **Business Notes & Best Practices**

- **Forecasting only works with “enough” data!** Use the history flag for quality control.
- **Safety stock is adaptive:** Higher for volatile or “critical” SKUs, lower for stable ones.
- **Scenario analysis** lets you communicate risk and cash trade-offs to stakeholders.
- **Flag slow or obsolete SKUs early** to avoid holding dead inventory.

---

## ✨ **Extending This Project**

- Plug in new demand models or time series tools.
- Add SKU/warehouse attributes for ABC analysis.
- Integrate with ordering APIs or warehouse management systems.
- Build dashboards from results_df for ongoing monitoring.

---

## 📚 **References**

- [Facebook Prophet documentation](https://facebook.github.io/prophet/)
- [Inventory control & safety stock theory](https://en.wikipedia.org/wiki/Safety_stock)
- [Normal distribution & z-scores](https://en.wikipedia.org/wiki/Standard_score)

---

## 📝 **Contact**

Questions? Ideas?  
Open an issue or reach out to the authors!


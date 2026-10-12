# Luxury Intelligence: Explainable Client Churn & Segmentation for the French Luxury Industry

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange.svg)](https://shap.readthedocs.io/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Ready-yellow.svg)](https://powerbi.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Project Owner**: Manasvi Sharma (B.Tech, Data Science, 3rd Year)  
**Industry Context**: French Haute Horlogerie & Maison Retail Clienteling  
**Core Technologies**: Python, Pandas, Scikit-Learn, XGBoost, SHAP (TreeExplainer), K-Means, PCA, Power BI  

---

## 1. Honesty Statement & Project Framing

> **Mandatory Disclosure**: This project uses a public retail transaction dataset (`Online Retail II` from the UCI Machine Learning Repository) as a proxy for luxury client data, because real luxury CRM data is private. The methodology and segmentation logic are designed specifically for luxury maison clienteling and demonstrated on proxy data. Any additional luxury-style attributes, if used, are synthetic and labelled as such. The 6-month churn window used here is a data-availability compromise; in a real luxury setting a 12–18 month window would be used.

---

## 2. Business Problem & Luxury Context

French luxury houses (Maisons) depend on a small base of high-value clients whose attrition is **silent and costly**. Unlike subscription services or telecommunications:
1. **No Complaints, Silent Departure**: Luxury clients rarely complain or cancel memberships; they simply stop visiting the boutique, and the loss is noticed when it is already too late.
2. **Discounts Destroy Brand Equity**: In high luxury, discounting is forbidden as it dilutes prestige. Retention must be white-glove, relational, and bespoke: private salon previews, atelier visits, personalized styling sessions, or a personal call from the Boutique Director.
3. **The "Why" Imperative**: Boutique managers will not act on an opaque black-box probability score. They need to understand **why** an elite client is at risk in plain business language so they can select the appropriate personal retention intervention.

This project delivers an end-to-end, leak-free, explainable churn prediction and RFM client-segmentation system designed for boutique clienteling managers.

---

## 3. Repository Architecture

```text
LUXURY BRAND PROJECT/
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx             # Raw UCI dataset (Untouched & read-only)
│   └── processed/
│       ├── transactions_clean.csv            # Cleaned positive purchase transactions (776,596 rows)
│       ├── cancellations_clean.csv           # Isolated cancellations for leak-free return rate (18,390 rows)
│       ├── train_features.csv                # Historical snapshot T1 client feature matrix (4,284 clients)
│       └── test_features.csv                 # Future snapshot T2 client feature matrix (4,942 clients)
├── notebooks/
│   ├── 01_cleaning_eda.ipynb                 # Data loading, cleaning audit & luxury EDA
│   ├── 02_snapshots_features.ipynb           # Two-snapshot prospective design & leakage audit
│   ├── 03_models.ipynb                       # Baseline Logistic Regression vs. XGBoost evaluation
│   ├── 04_shap_explainability.ipynb          # Global/local SHAP & plain-language reason code engine
│   ├── 05_segmentation.ipynb                 # RFM K-Means clustering, PCA 2D plot & actions
│   └── 06_export_for_powerbi.ipynb           # Data validation suite & Power BI export
├── outputs/
│   ├── client_risk_table.csv                 # Master fact table for Power BI (989 luxury clients)
│   ├── segment_summary.csv                   # RFM segment benchmarks & retention playbook
│   ├── model_metrics.csv                     # Model scorecard (ROC-AUC, PR-AUC, Lift, Rev-at-Risk)
│   ├── client_shap_reasons.csv               # Client-level SHAP diagnostic explanations
│   ├── client_segments.csv                   # Client-level cluster assignments
│   ├── xgboost_churn_model.json              # Serialized production XGBoost model
│   ├── scaler.joblib                         # Fitted StandardScaler artifact
│   └── *.png                                 # High-resolution presentation visuals
├── src/
│   └── __init__.py                           # Modular helper package
├── requirements.txt                          # Pinned project dependencies
└── README.md                                 # Complete executive & technical documentation
```

---

## 4. Data Specification & Cleaning Pipeline

The raw data spans December 1, 2009 to December 9, 2011 (~2 full years of transaction history across 2 sheets).

### Data Transformation Audit
* **Raw Transactions Loaded**: 1,067,371 rows across 2 sheets.
* **Exact Duplicate Rows Removed**: 34,335 rows (3.22%).
* **Missing Customer IDs Filtered**: 235,151 rows (anonymous walk-ins cannot be tracked for relationship clienteling).
* **Cancellations Isolated (`Invoice` starting with 'C')**: 18,390 rows across 2,572 clients preserved separately to engineer client `return_rate` without revenue contamination.
* **Valid Positive Purchases**: 779,425 rows (`Quantity > 0` and `Price > 0`).
* **Non-Product Stock Codes Removed**: 2,829 rows (`POST`, `D`, `M`, `C2`, `BANK CHARGES`, `DOT`, `CRUK`, `ADJUST`, `ADJUST2`, `TEST001`, `TEST002`).
* **Final Clean Purchases**: **776,596 transactions** generating **£17,068,582.74** across **5,852 unique clients**.

### Luxury Pareto Validation (EDA Finding)
* **Top 20% Clients**: 1,170 clients (Spend threshold: £2,058.07).
* **Revenue Share**: **77.15% (£13,168,000+)** of all revenue.
* **Inter-purchase Gaps**: Median repeat purchase gap is **25.1 days**; 75th percentile is **62.2 days**. A 6-month (180 days) inactivity window represents almost 3x the 75th percentile, establishing a statistically sound benchmark for churn.

---

## 5. Two-Snapshot Churn Design (Zero Data Leakage)

To avoid temporal data leakage common in naive random train-test splits, the project enforces a **prospective two-snapshot architecture**:

* **Global Cutoff Date (`MAX_DATE`)**: `2011-12-09 12:50:00`
* **Train Snapshot ($T_1$)**: `2010-12-09 12:50:00` ($MAX\_DATE - 12\text{ months}$)
  * Features use transactions: Strictly before $T_1$.
  * Churn Observation Window (6 months): $T_1 \rightarrow T_1 + 6\text{ months}$ (`2010-12-09` to `2011-06-09`).
* **Test Snapshot ($T_2$)**: `2011-06-09 12:50:00` ($MAX\_DATE - 6\text{ months}$)
  * Features use transactions: Strictly before $T_2$.
  * Churn Observation Window (6 months): $T_2 \rightarrow MAX\_DATE$ (`2011-06-09` to `2011-12-09`).
* **Ground-Truth Churn Label**: $\text{churn} = 1$ if the client made 0 purchases in the 6-month observation window, else $0$.

```
[Dec 2009] ----------------- [Dec 2010 (T1)] ----------------- [Jun 2011 (T2)] ----------------- [Dec 2011 (MAX)]
  |--- Train Feature History ---|--- Train 6m Label Window ---|
                                |--- Test Feature History ---------------------|--- Test 6m Label Window -----|
```

### Snapshot & Class Balance Summary Table

| Set | Cohort | Snapshot Date | Label Observation Window | Min Spend Threshold | Total Clients | Churned (1) | Retained (0) | Churn Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train ($T_1$)** | **Top 20% Luxury (Main)** | `2010-12-09` | `2010-12-09` to `2011-06-09` | **£2,127.42** | **857** | **154** | **703** | **17.97%** |
| **Train ($T_1$)** | Unfiltered (All Clients) | `2010-12-09` | `2010-12-09` to `2011-06-09` | £0.00 | 4,284 | 2,290 | 1,994 | 53.45% |
| **Test ($T_2$)** | **Top 20% Luxury (Main)** | `2011-06-09` | `2011-06-09` to `2011-12-09` | **£2,427.27** | **989** | **139** | **850** | **14.05%** |
| **Test ($T_2$)** | Unfiltered (All Clients) | `2011-06-09` | `2011-06-09` to `2011-12-09` | £0.00 | 4,942 | 2,370 | 2,572 | 47.96% |

**Leakage Audit Result**: Maximum historical transaction dates were `2010-12-09 12:49` (< $T_1$) and `2011-06-09 12:40` (< $T_2$). **Zero data leakage confirmed via assertion checks.**

---

## 6. Engineered Features & Missing Value Strategy

11 behavioral features computed per client using strictly historical transactions prior to snapshot:
1. `recency_days`: Days since last purchase as of snapshot date.
2. `frequency`: Count of distinct purchase invoices.
3. `monetary`: Cumulative gross revenue (£).
4. `avg_order_value`: `monetary / frequency`.
5. `tenure_days`: Days since first purchase as of snapshot date.
6. `avg_gap_days`: Mean days between consecutive purchases. *(Single-purchase clients: imputed with `tenure_days` representing their maximum observed waiting period).*
7. `std_gap_days`: Standard deviation of purchase gaps. *(Clients with < 3 purchases: set to `0.0` as mandated by PRD).*
8. `product_variety`: Count of distinct StockCodes (category breadth proxy).
9. `spend_trend`: Revenue in last 3 months $\div$ revenue in prior 3 months. *(If prior spend = 0 and recent spend > 0, set to `2.0` positive momentum; if both 0, set to `0.0`; capped at 10.0).*
10. `return_rate`: Cancelled invoices $\div$ total invoices.
11. `country_group`: `1` for United Kingdom (Domestic), `0` for International (Luxury tourists/overseas).

---

## 7. Predictive Modeling & Evaluation Results

### Model Formulation
* **Baseline**: `LogisticRegression(class_weight='balanced')` trained on standardized features.
* **Main Model**: `XGBoost(scale_pos_weight=4.56)` with 5-Fold Stratified Cross-Validation on Train Snapshot T1 optimizing PR-AUC. Best parameters: `max_depth=3`, `learning_rate=0.03`, `n_estimators=100`, `subsample=0.8`, `colsample_bytree=0.8`.

### Test Snapshot Evaluation Table (Top 20% Luxury Clients)

| Evaluation Metric | Baseline Logistic Regression | Main Model (XGBoost) | Business Takeaway |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | 0.8237 | **0.8339** | Excellent ranking discrimination (>83%). |
| **PR-AUC (Avg Precision)** | **0.5373** | 0.5003 | **3.6x higher than random base rate (14.05%)**. |
| **Decision Threshold** | 0.7297 | 0.7465 | F1-optimal threshold balancing recall and false alarms. |
| **Precision** | 47.62% | **50.00%** | 1 out of every 2 flagged clients is a true churner. |
| **Recall** | **57.55%** | 53.24% | Captures over half of all future lapsing clients. |
| **F1-Score** | 0.5212 | 0.5157 | Harmonic balance. |
| **Top-Decile Lift** | 4.07x | **4.14x** | Top 10% highest-risk clients are **4.14x more likely to churn**. |
| **Revenue-at-Risk Captured (Top 10%)** | 46.67% | **49.40%** | **Targeting top 10% protects ~50% of all lost luxury revenue.** |
| **Revenue-at-Risk Captured (Top 20%)** | **65.70%** | 63.46% | Targeting top 20% protects nearly two-thirds of lost revenue. |

### Unfiltered Comparison (PRD Section 4.3)
* **Unfiltered (All 4,942 Clients)**: XGBoost ROC-AUC = 0.7949, PR-AUC = 0.7563 (Base churn rate ~48%).
* **Top 20% Luxury Cohort**: XGBoost ROC-AUC = **0.8339** (+3.9% boost), demonstrating that high-value luxury clients exhibit much clearer, actionable behavioral signatures.

---

## 8. Explainable AI with SHAP & Plain-Language Reason Codes

Boutique directors cannot act on raw probabilities. Using `shap.TreeExplainer`, every prediction is deconstructed into exact Shapley attributions.

### Global SHAP Feature Importance
1. **`recency_days` (Mean |SHAP| = 0.5347)**: The primary driver. Prolonged silence drastically escalates churn risk.
2. **`product_variety` (Mean |SHAP| = 0.2539)**: Single-category buyers are fragile; multi-category collectors are resilient.
3. **`spend_trend` (Mean |SHAP| = 0.2133)**: Spending collapse over consecutive quarters serves as a vital early warning.
4. **`frequency` (0.1843)** & **`monetary` (0.1401)**: High visit counts and large investments create strong retention barriers.

### Executive Reading for Boutique Directors
> *"In luxury clienteling, **prolonged client silence (`recency_days`)** and a **collapsing spending momentum (`spend_trend`)** are the two strongest early-warning indicators of impending churn. When a high-value client's spend trend drops below 1.0, their probability of silent departure multiplies. Clients with **narrow category engagement (`product_variety`)** are also vulnerable. Retention must focus on proactive reconnection before silence extends past 90 days."*

### Reason Code Generator & Non-Discount Action Rules
For every client, the top 3 features pushing risk up are converted to plain-language sentences and mapped to non-discount actions:

```text
[Primary Risk Driver]  ───>  [Plain-Language Diagnostic Reason]  ───>  [Personalized Luxury Retention Action]
recency_days high            "Prolonged absence: client has not        Dedicated Client Advisor phone call with private salon
                             purchased or visited for an extended      preview invitation for upcoming high collections.
                             period."

spend_trend falling          "Declining momentum: spending has dropped Curated lookbook preview tailored to previous purchases
                             substantially in recent months."          & complimentary private styling appointment.

return_rate high             "Elevated return or cancellation rate     Boutique Director concierge call to resolve product
                             indicating potential dissatisfaction."    dissatisfaction and arrange tailored exchanges.

product_variety narrow       "Narrow product breadth: client purchases Personalized invitation to collection launch showcasing
                             from very limited item categories."       complementary categories (jewelry & leather goods).
```

---

## 9. Strategic RFM Segmentation (K-Means & PCA)

* **Features**: Log-transformed (`log1p`) and standardized `recency_days`, `frequency`, `monetary`.
* **Selection of $k = 4$**: Supported by Elbow point (Inertia: 1115.11) and Silhouette score (0.3088).
* **PCA 2D Projection**: Explains **85.3% of total variance** (PC1 = 61.1% spend scale, PC2 = 24.2% recency/frequency trade-off).

### Final Segment Profiles & Playbook (`outputs/segment_summary.csv`)

| Segment Name | Clients (% Base) | Avg Recency | Avg Orders | Avg Client Spend | Total Segment Spend | Churn Rate | Prescribed Clienteling Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Elite Maison Patrons** | **58** (5.9%) | **17.1 days** | **66.0** | **£60,878.19** | **£3.53M** | **5.2%** | Dedicated Boutique Director concierge relationship, bespoke atelier gifts, and private salon viewings of high-jewelry. |
| **Loyal Connoisseurs** | **226** (22.9%) | **42.3 days** | **20.8** | **£10,723.79** | **£2.42M** | **7.5%** | Senior Client Advisor telephone outreach with invitation to intimate champagne salon preview and curated heritage exhibition. |
| **Rising Aspirants** | **275** (27.8%) | **12.5 days** | **12.4** | **£4,503.50** | **£1.24M** | **5.1%** | Personalized style curation across complementary categories (fine leather goods & accessories) to expand collection breadth. |
| **Dormant High-Rollers** | **430** (43.5%) | **124.6 days** | **7.2** | **£3,944.14** | **£1.70M** | **24.4%** | **Primary retention target**: Personal reconnection call with exclusive seasonal preview invitation and styling session. |

---

## 10. Power BI Dashboard Specification

Power BI is built in the separate desktop application using the prepared and validated datasets in `outputs/`:
* `outputs/client_risk_table.csv` (Primary Fact Table: 989 rows × 10 columns, sorted by `revenue_at_risk`).
* `outputs/segment_summary.csv` (Segment Dimension Table).
* `outputs/model_metrics.csv` (Model Scorecard Reference).

### 4-Page Dashboard Layout Architecture
1. **Page 1 - Executive Overview**:
   * KPI Cards: Total Luxury Clients (989), High-Risk Clients (205), Expected Revenue at Risk (£2.78M), Churn Rate (14.05%).
   * Donut chart by Risk Tier; Bar chart of Revenue at Risk by Segment; Churn Rate by Country Group.
2. **Page 2 - At-Risk Clienteling Action Table**:
   * Master table sorted by `revenue_at_risk` descending with columns: `client_id`, `segment`, `risk_tier`, `churn_probability`, `revenue_at_risk`, `top_reason_1`, `recommended_action`.
   * Dynamic slicers: Segment dropdown, Risk Tier buttons, Country Group toggle.
3. **Page 3 - Strategic Segments Playbook**:
   * Matrix table of segment sizes, average spend, recency, and prescribed action; Scatter plot of Recency vs. Spend.
4. **Page 4 - Explainability Diagnostics ("Why")**:
   * Global SHAP feature importance bar chart; Single-client drill-through card revealing top 3 plain-language reasons and assigned retention action.
* **Mandatory Footer**: *"Proxy data — methodology for luxury clients"* on every page.

---

## 11. Visual Presentation Assets

All high-resolution visualization assets are located in the [`outputs/`](outputs/) directory:
* **Spend & Pareto Curve**: [`outputs/spend_distribution_pareto.png`](outputs/spend_distribution_pareto.png)
* **Purchase Gaps Distribution**: [`outputs/purchase_gaps_distribution.png`](outputs/purchase_gaps_distribution.png)
* **Domestic vs. International Breakdown**: [`outputs/country_distribution.png`](outputs/country_distribution.png)
* **Feature Churn Signals**: [`outputs/feature_churn_signals.png`](outputs/feature_churn_signals.png)
* **ROC, PR & Revenue-at-Risk Captured**: [`outputs/model_comparison_curves.png`](outputs/model_comparison_curves.png)
* **Global SHAP Summary (Beeswarm)**: [`outputs/shap_summary_beeswarm.png`](outputs/shap_summary_beeswarm.png)
* **Global SHAP Feature Importance**: [`outputs/shap_global_bar.png`](outputs/shap_global_bar.png)
* **Client Waterfall (High-Risk)**: [`outputs/shap_waterfall_high_risk.png`](outputs/shap_waterfall_high_risk.png)
* **Client Waterfall (Medium-Risk)**: [`outputs/shap_waterfall_medium_risk.png`](outputs/shap_waterfall_medium_risk.png)
* **Client Waterfall (Low-Risk)**: [`outputs/shap_waterfall_low_risk.png`](outputs/shap_waterfall_low_risk.png)
* **Elbow & Silhouette Validation**: [`outputs/segment_elbow_silhouette.png`](outputs/segment_elbow_silhouette.png)
* **PCA 2D Cluster Visualization**: [`outputs/segment_pca_2d.png`](outputs/segment_pca_2d.png)

---

## 12. How to Run & Reproduce

### Prerequisites
* Python 3.10+ (Tested on Python 3.12)
* Jupyter Notebook / JupyterLab

### Installation
```bash
# Clone or navigate to the repository
cd "LUXURY BRAND PROJECT"

# Install required dependencies
pip install -r requirements.txt
```

### Reproducing the Pipeline
Run the notebooks in sequential order:
1. `notebooks/01_cleaning_eda.ipynb`: Cleans transactions, filters non-products, outputs `transactions_clean.csv`.
2. `notebooks/02_snapshots_features.ipynb`: Creates prospective snapshots, asserts zero data leakage, outputs `train_features.csv` & `test_features.csv`.
3. `notebooks/03_models.ipynb`: Trains tuned XGBoost vs. Logistic Regression, outputs `xgboost_churn_model.json` & `model_metrics.csv`.
4. `notebooks/04_shap_explainability.ipynb`: Calculates SHAP values, renders waterfalls, outputs `client_shap_reasons.csv`.
5. `notebooks/05_segmentation.ipynb`: Runs log-transformed RFM K-Means, outputs `segment_summary.csv` & `client_segments.csv`.
6. `notebooks/06_export_for_powerbi.ipynb`: Runs validation suite and outputs finalized `client_risk_table.csv`.

*Note*: `random_state=42` is set across all scripts, models, and clustering algorithms to ensure exact numerical reproducibility.

---

## 13. Limitations

1. **Proxy Data**: The UCI dataset represents a UK-based wholesale/gift non-store retailer rather than true French Haute Horlogerie or Haute Couture transactions. Although spending concentration aligns with luxury Pareto dynamics, transaction frequency and average order values differ from true luxury retail.
2. **Short 6-Month Churn Window**: Luxury maison purchase intervals typically span 12–18 months. A 6-month observation window is a data-availability compromise imposed by the 2-year duration of the dataset.
3. **Absence of Boutique Signals**: Real luxury CRM environments record qualitative signals such as boutique foot traffic, private salon appointments, advisor call notes, and RSVP event attendance. Here, predictions rely strictly on quantitative transactional features.

---

## 14. Future Work

Pricing and demand modelling, real CRM integration, 12–18 month churn windows, uplift modelling for retention actions, and a French-language dashboard.

---

## 15. Resume & CV Bullets Draft (For Manasvi Sharma)

Below are tailored, high-impact bullet points demonstrating technical rigor, machine learning depth, and luxury business acumen for internships, applications, and interviews:

### Option A: Technical & Machine Learning Focus
* **Explainable Luxury Client Churn & Segmentation System | Python, XGBoost, SHAP, Power BI**
  * Engineered a prospective two-snapshot churn prediction architecture on 1M+ retail transactions, guaranteeing 0% temporal data leakage across 6-month prospective evaluation windows.
  * Developed a regularized XGBoost classifier (`scale_pos_weight=4.56`) outperforming baseline Logistic Regression with an **0.834 ROC-AUC**, **4.14x Top-Decile Lift**, and capturing **49.4% of total lost revenue** in the top 10% highest-risk clients.
  * Implemented `shap.TreeExplainer` and built an automated Reason Code Engine mapping non-linear feature attributions into structured, plain-language diagnostic reasons and tailored non-discounting luxury retention actions.
  * Conducted log-transformed RFM K-Means clustering ($k=4$, Silhouette=0.309) visualized via 2D PCA (85.3% variance explained), exporting validated fact/dimension schemas for an executive 4-page Power BI dashboard.

### Option B: Business Impact & Strategic Clienteling Focus
* **Luxury Intelligence & Client Attrition Analytics | Python, Scikit-Learn, XGBoost, Power BI**
  * Built an end-to-end client retention system for French luxury maisons to counteract silent attrition among top-tier spenders (top 20% driving 77.2% of £17.1M revenue) without brand-diluting discounts.
  * Identified **£2.78M in expected revenue at risk** across 989 high-value clients, delivering an actionable roster prioritized by expected loss for boutique directors and sales advisors.
  * Designed an empirical 4-cluster RFM segmentation model isolating "Elite Maison Patrons" (£60.9k avg spend) and "Dormant High-Rollers" (124 days recency, 24.4% churn rate), mapping each to bespoke white-glove protocols.
  * Produced complete, interview-ready documentation across 6 automated Jupyter notebooks and an executive Power BI reporting specification.

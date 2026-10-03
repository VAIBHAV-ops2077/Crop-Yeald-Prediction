# 🌾 Crop Yield Prediction Using Machine Learning

> **Course:** Machine Learning Fundamentals — Semester Project  
> **Case Study:** 29 — Crop Yield Prediction  
> **Tech Stack:** Python 3 · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Flask · Jupyter  

---

## 📋 Table of Contents

1. [Project Title & Problem Statement](#1-project-title--problem-statement)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [Why These Features Were Selected](#4-why-these-features-were-selected)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Machine Learning Algorithms](#7-machine-learning-algorithms)
8. [Model Training](#8-model-training)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Model Comparison](#10-model-comparison)
11. [Fertilizer vs Yield Analysis](#11-fertilizer-vs-yield-analysis)
12. [Residual Analysis](#12-residual-analysis)
13. [Feature Importance](#13-feature-importance)
14. [Best Model Selection](#14-best-model-selection)
15. [Web Application](#15-web-application)
16. [Project Structure](#16-project-structure)
17. [Installation & Running Instructions](#17-installation--running-instructions)
18. [Limitations](#18-limitations)
19. [Future Improvements](#19-future-improvements)
20. [Viva Questions / Important Concepts](#20-viva-questions--important-concepts)

---

## 1. Project Title & Problem Statement

**Crop Yield Prediction Using Machine Learning**

Agriculture is one of the most data-intensive industries in the world, yet most yield prediction is still done using rule-of-thumb or experience-based methods. Machine learning offers the ability to build quantitative models that learn patterns from historical records of farming conditions and outcomes.

**The Problem:** Given information about a farm's climate, soil, crop type, irrigation method, and farming practices, predict the **crop yield in metric tons per hectare (t/ha)**.

Accurate yield prediction helps:
- Farmers plan harvests and resource allocation in advance.
- Agricultural planners estimate food security risks.
- Researchers understand which environmental or management factors most influence yield.

---

## 2. Objectives

1. Load and explore the Agriculture Precision Study Cases dataset.
2. Select relevant input features and exclude leakage-prone or irrelevant columns.
3. Build a reproducible preprocessing pipeline that handles missing values and encodes categorical features correctly.
4. Train and evaluate **five regression algorithms** covering a range of model complexity.
5. Compare all models using R², MSE, RMSE, MAE, and 5-fold cross-validation.
6. Identify the best-performing model from actual experimental results.
7. Investigate the fertilizer–yield relationship (linear vs. nonlinear).
8. Analyse residuals and feature importances.
9. Deploy the best model as an interactive web application.

---

## 3. Dataset

**Dataset Name:** Agriculture Precision Study Cases — Modern Agriculture AI Crop Production 2020–2050  
**File:** `modern_agriculture_ai_crop_production_2020_2050.csv`  
**Source:** Kaggle (Agriculture Precision Study)  
**Total Records:** 75,000 rows × 36 columns  
**Target Variable:** `yield_ton_per_ha` — crop yield in metric tons per hectare  

### All 36 Columns — What They Mean & Why We Include or Exclude Each

| Column | Type | Range / Values | Used? | Reason |
|---|---|---|:---:|---|
| `record_id` | Integer | 1–75000 | ❌ | Row identifier — contains no predictive information |
| `year` | Integer | 2020–2050 | ❌ | Uncorrelated with yield; correlation = 0.01 |
| `region` | Categorical | 9 global regions | ✅ | Captures geographic / climatic zone differences |
| `country` | Categorical | 36 countries | ❌ | Too granular; region already captures geography at useful level |
| `crop` | Categorical | 10 varieties | ✅ | **Dominant predictor** — each crop has a distinct yield range |
| `farm_size_hectares` | Numeric | — | ❌ | Uncorrelated with yield *rate* (tons/ha) |
| `farm_type` | Categorical | 6 types | ✅ | Technology level of farm affects management practices |
| `soil_type` | Categorical | 6 types | ✅ | Soil texture affects water retention and nutrient availability |
| `irrigation_type` | Categorical | 6 types | ✅ | Irrigation method is a key agronomic management decision |
| `ai_system_used` | Categorical | — | ❌ | ~20% missing; outside scope of Case Study 29 |
| `technology_stack` | Categorical | — | ❌ | AI/tech variable — outside Case Study 29 scope |
| `ai_adoption_score` | Numeric | — | ❌ | AI metric — not a physical agronomic input |
| `mechanization_index` | Numeric | — | ❌ | Corr with yield = 0.02 — negligible |
| `iot_sensor_density_per_ha` | Numeric | — | ❌ | AI/tech — not an agronomic input |
| `drone_usage_hours_per_month` | Numeric | — | ❌ | AI/tech — not an agronomic input |
| `satellite_monitoring_score` | Numeric | — | ❌ | AI/tech — not an agronomic input |
| `avg_temperature_c` | Numeric | 7.7–46.9 °C | ✅ | Core climate driver of photosynthesis and crop growth |
| `annual_rainfall_mm` | Numeric | 50–1907 mm | ✅ | Primary water supply for rainfed agriculture |
| `soil_moisture_percent` | Numeric | 5–87.5% | ✅ | Reflects actual water available to roots |
| `fertilizer_kg_per_ha` | Numeric | 15–360 kg/ha | ✅ | Key management input for nutrient supply |
| `pesticide_litre_per_ha` | Numeric | 0.1–6.85 L/ha | ✅ | Pest control input — explicitly required by Case Study 29 |
| `water_use_m3_per_ha` | Numeric | — | ❌ | Derived from irrigation amount — potential leakage; corr=0.45 |
| `climate_stress_index` | Numeric | 0–100 | ✅ | Composite heat+drought stress score — strongest numeric predictor |
| `disease_risk_score` | Numeric | — | ❌ | Corr with yield = 0.03; adds noise without signal |
| `pest_risk_score` | Numeric | — | ❌ | Corr with yield = 0.03; adds noise without signal |
| `yield_ton_per_ha` | Numeric | 1.07–105.91 | **TARGET** | What we predict |
| `production_tons` | Numeric | — | ❌ | **Leakage** — production = yield × farm_size |
| `operational_cost_usd_per_ha` | Numeric | — | ❌ | Financial output, not an input; corr = 0.03 |
| `market_price_usd_per_ton` | Numeric | — | ❌ | Market variable, uncorrelated with yield |
| `revenue_usd` | Numeric | — | ❌ | **Leakage** — derived from yield × price × area |
| `profit_usd` | Numeric | — | ❌ | **Leakage** — derived from revenue − cost |
| `carbon_emission_kgco2e_per_ha` | Numeric | — | ❌ | Consequence of farming practice, not an input |
| `sustainability_score` | Numeric | — | ❌ | Corr with yield = 0.21; composite score, not an agronomic input |
| `yield_forecast_next_year_ton_per_ha` | Numeric | — | ❌ | **Critical leakage** — corr with yield = 0.998 |
| `risk_category` | Categorical | 4 levels | ❌ | Label derived from yield/stress — **leakage** |
| `recommended_action` | Categorical | 6 actions | ❌ | Label derived from other features — **leakage** |

### Selected Features (11 inputs + 1 target)

**Numerical Features (6):**
- `avg_temperature_c` — average seasonal temperature
- `annual_rainfall_mm` — total annual precipitation
- `fertilizer_kg_per_ha` — fertilizer applied
- `pesticide_litre_per_ha` — pesticide applied
- `soil_moisture_percent` — volumetric soil moisture
- `climate_stress_index` — composite drought/heat stress (0=no stress, 100=extreme)

**Categorical Features (5):**
- `crop` — crop variety (10 options)
- `soil_type` — soil texture (6 options)
- `irrigation_type` — irrigation method (6 options)
- `region` — geographic zone (9 options)
- `farm_type` — farming technology level (6 options)

**Target (1):**
- `yield_ton_per_ha` — crop yield in tons per hectare

### Basic Statistics of Selected Numerical Features

| Feature | Mean | Std | Min | Max |
|---|:---:|:---:|:---:|:---:|
| `avg_temperature_c` | 27.0 °C | 5.5 | 7.7 | 46.9 |
| `annual_rainfall_mm` | 651 mm | 276 | 50 | 1907 |
| `fertilizer_kg_per_ha` | 135 kg/ha | 54 | 15 | 360 |
| `pesticide_litre_per_ha` | 2.4 L/ha | 1.1 | 0.1 | 6.85 |
| `soil_moisture_percent` | 36.8% | 11.2 | 5.0 | 87.5 |
| `climate_stress_index` | 46.8 | 15.3 | 0 | 100 |
| `yield_ton_per_ha` | 17.1 t/ha | 21.3 | 1.07 | 105.91 |

---

## 4. Why These Features Were Selected

The feature selection was guided by three principles:

**1. Avoid Data Leakage:**  
Columns like `production_tons` (= yield × area), `revenue_usd`, `profit_usd`, and most critically `yield_forecast_next_year_ton_per_ha` (correlation = 0.998 with target) were removed. Using these in training would artificially inflate performance because they are derived from or directly contain yield information.

**2. Agronomic Relevance:**  
The kept features are genuine pre-harvest inputs: climate conditions (temperature, rainfall, stress), soil properties (type, moisture), and management decisions (fertilizer, pesticide, irrigation, farm type). These are what a farmer or agronomist would know *before* harvest.

**3. Statistical Signal:**  
AI/technology-related columns (drone hours, IoT density, satellite score) had correlations near 0.01–0.02 with yield and are outside the agricultural scope of Case Study 29. They were excluded to keep the model focused and interpretable.

---

## 5. Data Preprocessing

All preprocessing follows strict **anti-leakage discipline**: every transformer is fitted **only on training data**, then applied to test data.

### 5.1 Missing Value Handling
- Only one column has missing values: `ai_system_used` (~20% missing), which is excluded.
- Among the 11 selected features, there are **no missing values**.
- A `SimpleImputer(strategy='median')` is still included in the numerical pipeline as a defensive safeguard in case future data batches contain gaps.

### 5.2 Categorical Encoding — OneHotEncoder
- All 5 categorical features (`crop`, `soil_type`, `irrigation_type`, `region`, `farm_type`) are encoded using **`OneHotEncoder`**.
- **Why not LabelEncoder?** `LabelEncoder` converts categories to integers (e.g., Clay=0, Loam=1, Sandy=2), which introduces a false numerical ordering. For nominal categories, there is no inherent ordering, and using integers would cause linear models and distance-based methods to treat Clay as "less than" Sandy — which is meaningless. `OneHotEncoder` creates a binary column for each category, correctly representing them as mutually exclusive options.
- `handle_unknown='ignore'`: If the web app submits a category not seen during training, it is treated as all zeros instead of raising an error.

### 5.3 Train/Test Split
- 80% training set (60,000 samples)
- 20% test set (15,000 samples)
- `random_state=42` — ensures identical splits on every run

### 5.4 Pipeline Architecture
```
Input → ColumnTransformer ──┬── num_transformer (SimpleImputer) ──► Numerical columns
                             └── cat_transformer (Imputer + OHE)  ──► Categorical columns
                          └──────────────────────────────────────────► Model
```
The entire preprocessing + model is packaged as a single Scikit-learn `Pipeline`, which is saved to `models/best_model.pkl`. The Flask app loads this exact pipeline, so **prediction preprocessing is guaranteed to be identical to training preprocessing**.

---

## 6. Exploratory Data Analysis (EDA)

All plots are saved to `results/`. Key findings:

| Plot | File | Key Finding |
|---|---|---|
| Yield Distribution | `01_yield_distribution.png` | Multimodal distribution reflecting diverse crop types (Sugarcane ~69 t/ha vs Wheat ~3.6 t/ha) |
| Yield by Crop | `02_yield_by_crop.png` | **Crop variety is the dominant predictor** — each crop occupies a distinct yield range |
| Correlation Heatmap | `03_correlation_heatmap.png` | Climate stress index has the strongest negative correlation with yield numerically |
| Rainfall vs Yield | `04_rainfall_vs_yield.png` | Near-zero overall correlation — hidden by crop-variety variance |
| Temperature vs Yield | `05_temperature_vs_yield.png` | Weak direct correlation; temperature interacts with crop type |
| Fertilizer vs Yield | `06_fertilizer_vs_yield_fit.png` | See Section 11 below |
| Pesticide vs Yield | `07_pesticide_vs_yield.png` | Near-zero marginal correlation — crop type dominates |
| Climate Stress vs Yield | `08_climate_stress_vs_yield.png` | Clear negative trend — most informative single numerical feature |
| Soil & Irrigation | `09_categorical_vs_yield.png` | Near-equal yield means across soil and irrigation types (crop variety dominates) |

---

## 7. Machine Learning Algorithms

### Linear Regression

**What it is:** The simplest regression model. It fits a weighted combination of all input features:  
$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$  
Each weight $w_i$ represents how much yield changes for a unit increase in feature $x_i$, holding all others constant.

**Why it is used as a baseline:** If a simpler model performs nearly as well as a complex one, it suggests the relationships in the data are approximately linear and the simpler model is preferable for interpretability. On this dataset, Linear Regression achieves R² = 0.9710, which reveals that after proper OneHotEncoding of crop types, the remaining numeric variation is largely captured linearly.

**How it is applied here:** The OHE encoding of crop types effectively creates dummy variables for each crop, allowing linear regression to learn different intercepts per crop — which closely matches the biologically distinct yield ranges. Temperature, rainfall, fertilizer, and climate stress then contribute marginal adjustments around those baselines.

**Limitations:** Cannot model interaction effects (e.g., how the effect of rainfall depends on the crop) or nonlinear curves without manual feature engineering.

---

### Polynomial Regression (Degree 2)

**What it is:** An extension of Linear Regression that adds squared terms ($x_i^2$) and cross-product interaction terms ($x_i \cdot x_j$) for numerical features. These extra columns allow the model to fit curves and capture how combinations of features jointly affect yield.

**Why it is included:** Agronomic theory predicts nonlinear relationships — for example, crop yield may first increase with fertilizer application and then plateau due to diminishing returns (Mitscherlich's Law). Polynomial features let the model detect such curvature without switching to a completely different algorithm.

**How polynomial features capture nonlinear relationships:** A degree-2 term for fertilizer ($\text{fertilizer}^2$) allows the model to fit a parabola instead of a straight line. If the optimal fertilizer level exists, the parabola would peak there and decline.

**Why it is useful for fertilizer-vs-yield analysis:** Even when the overall trend appears flat, polynomial regression can detect weak curvature that linear regression would miss. In this dataset, it gives the same performance as linear regression, which is itself a finding — the data does not show a statistically significant nonlinear fertilizer response.

**How it is applied here:** Polynomial features (degree 2) are applied only to the 6 numerical columns, after scaling with `StandardScaler`. The 5 categorical features are OneHotEncoded separately and then joined. This avoids creating polynomial cross-products of binary OHE columns, which would be meaningless.

**Limitations:** The number of features grows rapidly with degree — degree 2 on 6 numerical features produces 27 additional columns. With large polynomial degrees or many features, it becomes unstable and prone to overfitting.

---

### Decision Tree Regressor

**How it works:** Recursively splits the training data into groups by asking yes/no questions about feature values (e.g., "Is the crop Sugarcane?"). Each split minimizes the variance of yield values within each resulting group. The final model assigns the mean yield of each leaf node as the prediction.

**Why it is useful for nonlinear agricultural relationships:** A Decision Tree can perfectly capture the fact that Sugarcane yields ~69 t/ha regardless of temperature variation, while Wheat yields ~3.6 t/ha — without requiring any linear assumption. It also naturally handles threshold effects (e.g., yield drops sharply when climate stress index exceeds 70).

**How it is trained in this project:** `max_depth=10` limits tree growth to prevent memorizing training noise. Each split is evaluated on all 11 features (after preprocessing).

**Advantages:** Easy to explain ("if Sugarcane and low stress → high yield"), handles nonlinearity, works with mixed feature types.

**Limitations:** A single tree is high-variance — small changes in training data can produce very different trees. Ensembles (Random Forest, Gradient Boosting) address this.

---

### Random Forest Regressor

**How it works:** Trains 100 Decision Trees, each on a randomly sampled subset of training rows (bootstrap sample) and at each split considers only a random subset of features. The final prediction is the **average** of all 100 trees' predictions.

**Why multiple decision trees are useful:** Each individual tree overfits slightly differently to its bootstrap sample. When we average their predictions, these errors cancel out, yielding a lower-variance, more stable model than any single tree.

**Why it can capture feature interactions:** Each tree's splits can encode combinations like "Sugarcane AND high rainfall AND low climate stress → very high yield" without explicit feature engineering.

**How it is applied:** `n_estimators=100`, `max_depth=15`, `random_state=42`, `n_jobs=-1` (parallel training).

**Advantages:** Highly accurate, naturally resistant to overfitting, provides feature importances.

**Limitations:** Less interpretable than a single tree; slower to predict than linear models at very large scale.

---

### Gradient Boosting Regressor

**How boosting works:** Trains trees **sequentially** rather than independently. The first tree makes a rough prediction. The second tree is trained specifically on the **residual errors** (mistakes) of the first. Each subsequent tree corrects the remaining errors of all previous trees. The final prediction is the weighted sum of all trees.

**How successive trees correct previous errors:** If the first tree predicts 20 t/ha for a Tomato record that actually yielded 38 t/ha, the second tree targets the error of −18 t/ha. By focusing on errors rather than re-learning the full target, boosting can refine predictions with shallow trees.

**Why it is suitable for this prediction problem:** The multi-modal yield distribution (10 distinct crop-level ranges with within-crop variation from climate and management factors) is well-suited to sequential residual correction. Gradient Boosting can handle the combination of crop-type effects (large jumps) and climate stress effects (small within-crop variation).

**How it is applied:** `n_estimators=150`, `learning_rate=0.1`, `max_depth=6`, `random_state=42`.

**Advantages:** Typically achieves the highest accuracy on tabular datasets; handles complex interactions.

**Limitations:** More hyperparameters to tune; slower to train than Random Forest; slightly harder to interpret.

---

## 8. Model Training

All five models are trained as complete Scikit-learn **Pipelines**:
```
Pipeline(steps=[
    ('pre',   ColumnTransformer([num_pipeline, cat_pipeline])),
    ('model', <algorithm>)
])
```

Packaging preprocessing and model together in a Pipeline provides several guarantees:
1. **No leakage:** `preprocessor.fit()` runs only on `X_train`.
2. **Flask consistency:** The same saved pipeline object handles both training and prediction, so the web app never applies different preprocessing than what was used during training.
3. **Cross-validation safety:** `cross_val_score` also applies the full pipeline (fit on each fold's training split), so CV scores are honest estimates.

---

## 9. Evaluation Metrics

Every model is assessed on the **held-out test set** (15,000 samples, never seen during training) using four metrics, plus 5-fold cross-validation on training data:

| Metric | Formula | What It Measures | Better When |
|---|---|---|---|
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | % of yield variance explained by the model | Closer to 1.0 |
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error in (t/ha)² | Lower |
| **RMSE** | $\sqrt{\text{MSE}}$ | Square root of MSE — in the same units as yield (t/ha) | Lower |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error in t/ha | Lower |
| **5-Fold CV R²** | Mean ± Std of R² across 5 folds | Generalization stability | High mean, low std |

**RMSE vs MAE — an important distinction:** MAE is the straightforward average of the absolute prediction errors — every error contributes equally. RMSE first squares each error before averaging, which means larger errors contribute disproportionately more than small ones. For example, an error of 6 t/ha contributes four times as much to RMSE as an error of 3 t/ha, but only twice as much to MAE. As a result, RMSE is always ≥ MAE, and a large gap between them indicates the model makes occasional large errors. In this project, RMSE = 3.59 t/ha and MAE = 1.80 t/ha — the gap reflects the presence of larger residuals in the high-yield crops (Sugarcane, Tomato) where absolute yield values are naturally larger.

---

## 10. Model Comparison

Results are generated from actual experimental execution. The winning model is selected based on the highest test R²:

| Rank | Model | R² | MSE | RMSE (t/ha) | MAE (t/ha) | 5-Fold CV R² |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **Random Forest** | **0.9717** | **12.8888** | **3.5901** | **1.7960** | **0.9727 ± 0.0005** |
| 2 | Gradient Boosting | 0.9716 | 12.9090 | 3.5929 | 1.7993 | 0.9726 ± 0.0005 |
| 3 | Linear Regression | 0.9710 | 13.2094 | 3.6345 | 1.9023 | 0.9723 ± 0.0004 |
| 4 | Polynomial Reg. (d=2) | 0.9710 | 13.2119 | 3.6348 | 1.9029 | 0.9723 ± 0.0004 |
| 5 | Decision Tree | 0.9672 | 14.9374 | 3.8649 | 1.8976 | 0.9683 ± 0.0003 |

> These are actual results from running the notebook, not pre-computed or hardcoded values.

> **Important disclaimer:** These metrics reflect model performance on this specific simulation-origin dataset and should **not** be interpreted as real-world agricultural prediction accuracy. R² ≈ 0.97 is substantially higher than typically achievable in real-world crop yield prediction, where confounding factors (unrecorded pest events, equipment failures, soil variability within fields, market-driven planting decisions) are present.

**Key observation:** All five models achieve R² > 0.96, which is unusual. This indicates that `yield_ton_per_ha` is strongly determined by crop type — each variety operates in a distinct, non-overlapping yield range. Once crop type is OneHotEncoded, even a linear model can explain most of the total variance. The practical differences in R² between models are very small (range: 0.9672–0.9717), and the top four models are separated by only 0.0045 in R².

---

## 11. Fertilizer vs Yield Analysis

**Requirement of Case Study 29:** Investigate whether the relationship between fertilizer quantity and yield is linear or nonlinear.

### What This Dataset Actually Shows

After analysis of both the full dataset and individual crop subsets:

1. **Overall correlation (all crops): Pearson r ≈ 0.002** — essentially zero. Crop variety variance (~10× yield difference between high- and low-yield crops) completely overwhelms any marginal fertilizer signal.

2. **Within-crop analysis (e.g., Wheat):** Pearson r ≈ 0.019 — still very weak. Within a single crop, fertilizer amount shows minimal linear relationship with yield.

3. **Linear vs. Polynomial fit:** Both a linear trend line and a degree-2 polynomial curve are nearly horizontal. The quadratic curve does **not** show the inverted-U (Mitscherlich) shape that theoretical agronomic models predict.

**Conclusion from the data:** The expected nonlinear fertilizer–yield relationship (diminishing returns, then yield reduction at excessive application) was **NOT detected** in this dataset. Both the linear and polynomial fits are approximately flat, and neither describes a meaningfully different relationship from the other.

**This is the correct conclusion to report.** Reporting a fabricated diminishing-returns curve — or claiming excessive fertilizer reduces yield — would misrepresent what the data actually shows. The flat trend is a genuine property of this dataset's structure, not a modelling failure.

**Why Polynomial Regression is still included and valuable:**
- It is a required algorithm for this case study.
- The near-equal performance of Polynomial and Linear Regression (R² = 0.9710 for both) is itself an informative finding — adding squared and cross-product terms brings no additional predictive power in this dataset.
- In other agricultural datasets where fertilizer is a primary driver, polynomial regression would be expected to outperform a linear baseline.

---

## 12. Residual Analysis

For the best model (Random Forest), residual analysis was performed:

**Residual Plot (`11_actual_vs_predicted_residuals.png`):**
- The scatter of residuals vs. predicted values is approximately random around zero for most predictions.
- **Mean residual: −0.050** (very close to zero, indicating an unbiased model).
- **Residual Std: 3.59** t/ha (matches RMSE, as expected).
- **Maximum error: 30.6 t/ha** — the largest single-record prediction error.

**Interpretation:**
- The random scatter pattern indicates the model is not systematically over- or under-predicting for any particular yield range.
- Large residuals (>15 t/ha) occur mostly in the high-yield range (Sugarcane, Tomato), where yield variation is naturally larger.
- There is a slight fan-shape (heteroscedasticity) visible at high predicted yields — this is expected because high-yield crops like Sugarcane (range: 38–106 t/ha) have larger absolute error magnitudes than low-yield crops like Cotton (range: 1–3.4 t/ha).

**What the residual behaviour means:** The model generalizes well and is not overfitting to training patterns. The primary source of remaining error is the natural biological variability within each crop type that is not captured by the measured features.

---

## 13. Feature Importance

Feature importances were extracted from the tree-based models (Random Forest and Gradient Boosting). Since categorical features are OneHotEncoded, the importance of each category appears as a separate feature name (e.g., `crop_Sugarcane`, `crop_Tomato`).

**Key findings from actual importances:**

| Rank | Feature | Type | Agronomic Interpretation |
|:---:|---|---|---|
| 1 | `crop_Sugarcane` | OHE | Sugarcane has uniquely high yield (~69 t/ha) — single largest discriminator |
| 2 | `crop_Tomato` | OHE | Second-highest yield group (~40 t/ha) |
| 3 | `crop_Potato` | OHE | Third-highest (~26 t/ha) |
| 4 | `climate_stress_index` | Numeric | Highest importance among numerical features — scores above this rank every numerical input |
| 5 | `soil_moisture_percent` | Numeric | Water availability at root level |
| 6 | `avg_temperature_c` | Numeric | Temperature affects photosynthesis rate |
| 7–10 | Other crop OHE + other numeric | Mixed | Smaller marginal contributions |

**Important disclaimers:**
- Feature importance shows **predictive association**, not causation.
- Crop type appears most important because it separates distinct yield ranges — this is a dataset structure effect, not necessarily a causal claim.
- `climate_stress_index` is the strongest numerical predictor in this model. However, the exact construction of this column in the source dataset is not documented — it is described as a composite of drought and heat indicators, but it is not possible to independently verify from the dataset alone that it contains no yield-derived information. This is noted as a dataset limitation (see Section 18).

---

## 14. Best Model Selection

The best model is selected **after running all experiments** based on the highest test R²:

**Winner: Random Forest Regressor**  
- R² = 0.9717 (highest on test set)
- RMSE = 3.59 t/ha
- MAE = 1.80 t/ha
- 5-Fold CV R² = 0.9727 ± 0.0005

**Why Random Forest over Gradient Boosting (R² = 0.9716)?**  
The R² gap is only **0.0001** — this is a negligible difference. Both models achieve near-identical RMSE (3.5901 vs 3.5929 t/ha), MAE (1.7960 vs 1.7993 t/ha), and 5-Fold CV R² (0.9727 vs 0.9726). Random Forest is reported as the winner strictly by metric ordering; in practice, **their predictive performance is essentially identical**. Either model would be a valid choice for deployment.

**Why not Linear Regression (R² = 0.9710)?**  
The gap over Linear Regression is also small (0.0007 in R²), which shows that OHE encoding of crop type does most of the predictive work regardless of model complexity. Random Forest is preferred because it provides feature importances and can capture any residual nonlinearity.

**Why not Decision Tree (R² = 0.9672)?**  
The single Decision Tree has a 0.0045 lower R² and 7.6% higher RMSE than Random Forest. Averaging 100 trees (bagging) reduces prediction variance and improves stability, making the ensemble approach clearly preferable to a single tree.

---

## 15. Web Application

A simple, student-understandable interactive web application is provided in `app/`.

**Architecture:**
- **Backend:** `app/app.py` — Flask routes that load the saved pipeline and serve predictions.
- **Frontend:** `app/templates/index.html` — HTML form with Jinja2 templating for dynamic dropdowns.
- **Styling:** `app/static/css/style.css` — Vanilla CSS.
- **Client Logic:** `app/static/js/main.js` — Fetch API call to POST inputs and display results.

**What the user inputs:**
- Average Temperature (°C)
- Annual Rainfall (mm)
- Soil Moisture (%)
- Climate Stress Index (0–100)
- Fertilizer Applied (kg/ha)
- Pesticide Applied (litres/ha)
- Crop Variety (10 options)
- Soil Type (6 options)
- Irrigation Type (6 options)
- Region (9 options)
- Farm Type (6 options)

**What the app displays:**
- Predicted Crop Yield (tons/ha)
- Expected Prediction Error (±RMSE) — labeled clearly as an error range, not a confidence interval
- Model name
- Test R² and RMSE

**Error range labeling:**  
The displayed range is `Predicted Yield ± RMSE`. RMSE is the root mean squared error from the held-out test set. This is **not a formal statistical confidence interval** — it is the typical magnitude of prediction error observed on test data. About 65–70% of actual test predictions fell within ±RMSE of the true values.

---

## 16. Project Structure

```text
crop-yield-prediction/
├── app/
│   ├── app.py                          # Flask backend — loads pipeline, serves /predict
│   ├── predict.py                      # Standalone CLI prediction script
│   ├── templates/
│   │   └── index.html                  # Responsive web UI with Jinja2 templating
│   └── static/
│       ├── css/
│       │   └── style.css               # Vanilla CSS styling
│       └── js/
│           └── main.js                 # Fetch API client-side handler
├── data/
│   └── modern_agriculture_ai_crop_production_2020_2050.csv   # Kaggle dataset (75,000 rows)
├── models/
│   └── best_model.pkl                  # Saved pipeline artifact (preprocessor + model)
├── notebooks/
│   └── crop_yield_prediction.ipynb     # Fully executed Jupyter notebook
├── results/
│   ├── model_comparison.csv            # Benchmark table for all 5 models
│   ├── 01_yield_distribution.png
│   ├── 02_yield_by_crop.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_rainfall_vs_yield.png
│   ├── 05_temperature_vs_yield.png
│   ├── 06_fertilizer_vs_yield_fit.png
│   ├── 07_pesticide_vs_yield.png
│   ├── 08_climate_stress_vs_yield.png
│   ├── 09_categorical_vs_yield.png
│   ├── 10_r2_comparison.png
│   ├── 11_actual_vs_predicted_residuals.png
│   ├── 12_all_models_actual_vs_predicted.png
│   └── 13_feature_importance.png
├── requirements.txt
└── README.md
```

---

## 17. Installation & Running Instructions

### Prerequisites
- Python 3.9 or newer
- `pip` package manager

### Step 1: Install Dependencies
```bash
cd crop-yield-prediction
pip install -r requirements.txt
```

Dependencies installed: `Flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `jupyter`.

### Step 2: Run the Jupyter Notebook (Full Analysis)
```bash
jupyter notebook notebooks/crop_yield_prediction.ipynb
```
The notebook is pre-executed with all outputs. To re-run from scratch: **Kernel → Restart & Run All**.

### Step 3: Run the Web Application
```bash
python app/app.py
```
Open your browser at: **http://127.0.0.1:5000**

Enter the field parameters and click **"Predict Yield"** to see the predicted crop yield.

### Step 4: (Optional) CLI Predictor
```bash
python app/predict.py
```
Follow the prompts to enter field conditions and receive a yield prediction in the terminal.

---

## 18. Limitations

1. **Dataset is simulation-origin:** The Kaggle dataset, while structured around realistic crop and climate categories, uses computationally generated data. Performance metrics (R² ≈ 0.97) are substantially higher than typically observed in real-world agricultural prediction. In real settings, confounding factors (unrecorded pest events, soil variability within fields, equipment failures, market-driven planting decisions) introduce variation that no measured feature can fully capture. **R² = 0.9717 should not be presented as evidence of real-world prediction accuracy.**

2. **Crop variety dominates all other features:** Once crop type is OneHotEncoded, the model has effectively learned the mean yield per crop variety. Agronomic inputs (fertilizer, rainfall, irrigation) contribute minimal marginal variation. A more practically useful model would be trained separately per crop type, so that within-crop variation from climate and management factors becomes the primary signal.

3. **Fertilizer signal is negligible:** The expected nonlinear (diminishing returns) fertilizer–yield relationship was not detected in this dataset. Both linear and polynomial fits are approximately flat. This limits the practical insight from the fertilizer analysis.

4. **`climate_stress_index` construction unverifiable:** This feature is the strongest numerical predictor in the model, but its exact construction method is not documented in the dataset source. It is described as a composite of drought and heat indicators. Without access to the generation code, it cannot be confirmed that it contains no information derived from yield itself. This should be considered when interpreting its high feature importance.

5. **No temporal structure modeled:** The data spans 2020–2050 but year was excluded as uncorrelated with yield. Time-series effects (soil degradation, technology improvement over decades) are not captured.

6. **No geographic coordinates:** Using region as a categorical feature is a coarse approximation of the diverse climates within each of the 9 global regions.

---

## 19. Future Improvements

1. **Use real-world crop yield data** (FAOSTAT, USDA NASS, or state agricultural departments) to build models with genuine agronomic signal.
2. **Crop-stratified modeling:** Train separate models per crop type to focus prediction on within-crop yield variation from climate and management factors.
3. **Add temporal features:** Rolling average rainfall and temperature, cumulative growing degree days.
4. **Use SHAP values** instead of Gini-based feature importances for more reliable explanation of individual predictions.
5. **Hyperparameter tuning:** Apply `GridSearchCV` or `RandomizedSearchCV` to systematically optimize model configurations.
6. **Model calibration:** Use prediction intervals (e.g., quantile regression forests) to report statistically valid uncertainty ranges.

---

## 20. Viva Questions / Important Concepts

**Q: Why did you choose OneHotEncoder instead of LabelEncoder for categorical features?**  
A: LabelEncoder assigns integers (e.g., Clay=0, Loam=1, Sandy=2), which implies an artificial numeric ordering. For nominal categories like soil type or crop variety, there is no such ordering. OneHotEncoder creates a binary column for each category, letting the model treat them as mutually exclusive without imposing false numerical relationships.

**Q: What is data leakage, and what columns did you exclude because of it?**  
A: Data leakage occurs when information that would not be available at prediction time is used in training. We excluded `production_tons` (= yield × area), `revenue_usd`, `profit_usd`, and especially `yield_forecast_next_year_ton_per_ha` (correlation = 0.998 with the target) — all derived from or very strongly correlated with the yield we are trying to predict.

**Q: Why is R² so high (~0.97) across all models?**  
A: Crop variety creates large, discrete yield ranges (Sugarcane: ~69 t/ha, Cotton: ~2.2 t/ha). Once the model encodes crop type via OneHotEncoding, it essentially learns a per-crop mean yield, which explains most of the total variance. The numeric features (temperature, rainfall, etc.) add marginal refinement. This is a characteristic of the dataset structure, not evidence of real-world prediction power.

**Q: What is the difference between RMSE and MAE, and what does RMSE = 3.59 t/ha mean in plain English?**  
A: **MAE** (Mean Absolute Error) is the straightforward average of all absolute prediction errors — every error contributes proportionally to the total. **RMSE** (Root Mean Squared Error) first squares each error before averaging, which means a prediction error of 10 t/ha contributes 100 times as much as an error of 1 t/ha to the squared sum (rather than 10 times). Squaring penalises large errors disproportionately. As a result, RMSE is always ≥ MAE, and a large gap between them signals that the model occasionally makes very large errors — not just many small ones.

In this project, RMSE = 3.59 t/ha and MAE = 1.80 t/ha. The gap (approximately 2:1) indicates the model makes some larger residuals — particularly for high-yield crops (Sugarcane, Tomato) where the absolute yield values and natural biological variance are larger. For Wheat (mean 3.6 t/ha), a prediction error of ±3.59 t/ha represents a very high relative error (~100%). For Sugarcane (mean 69 t/ha), the same RMSE represents ~5% relative error. RMSE blends these heterogeneous error magnitudes across all crops into a single number.

**Q: What is the difference between Random Forest and Gradient Boosting?**  
A: Both use multiple decision trees. Random Forest trains trees **in parallel**, each on a bootstrap sample, and averages predictions (bagging). Gradient Boosting trains trees **sequentially** — each tree focuses specifically on correcting the prediction errors of all previous trees. Gradient Boosting is often slightly more accurate but more sensitive to hyperparameter choices.

**Q: Why is 5-fold cross-validation used in addition to a test set?**  
A: The test set gives one estimate of generalization. 5-fold CV uses different training-validation splits 5 times and averages the results, giving a more stable estimate of how much the model's performance varies with different data subsets. The standard deviation across folds (e.g., ±0.0005) tells us how consistent the model is.

**Q: What does the fertilizer analysis show, and is it what agronomic theory predicts?**  
A: In this dataset, fertilizer has near-zero correlation with yield (overall Pearson r ≈ 0.002). Both linear and polynomial fits are approximately flat. Agronomic theory (Mitscherlich's Law) predicts a diminishing-returns curve, but that signal is absent here — possibly because the dataset simulates a wide range of unrelated crop types and the fertilizer variable is distributed independently of yield. The conclusion comes from the data, not from theory.

**Q: What does a residual plot tell you?**  
A: A residual plot shows the difference (actual − predicted) on the y-axis against predicted values on the x-axis. If the model is well-specified, residuals should appear as a random horizontal band around zero (no systematic over/under-prediction). Patterns like a funnel shape (heteroscedasticity) or curves suggest the model is missing a relationship that the data contains.

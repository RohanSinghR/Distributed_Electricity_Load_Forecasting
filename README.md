# Distributed Electricity Load Forecasting using Spark and AWS EMR

A scalable, cloud-native Big Data pipeline for forecasting regional electricity consumption across multiple PJM power grid regions.
---

## Overview

Electricity load forecasting is a foundational requirement for efficient power system planning, operational stability, and cost-effective energy distribution. With smart meters and digital grid infrastructure generating massive volumes of high-frequency consumption data, traditional single-node systems can no longer keep up. This project addresses that challenge by designing and implementing a complete Big Data forecasting pipeline capable of handling millions of time-stamped electricity load records across multiple regions.

The pipeline integrates **Amazon S3** for distributed storage, **Amazon EMR** with **Apache Spark** for large-scale computation, and advanced machine learning algorithms for accurate next-hour demand prediction. The system covers the full workflow — from raw data ingestion and feature engineering through model training, evaluation, and automated visual reporting — across five PJM electricity regions: AEP, COMED, DAYTON, DOM, and PJM.

---

## Architecture

```
Amazon S3 (Parquet Data Lake)
  └── region=AEP / COMED / DAYTON / DOM / PJM
        └── year → month → parquet files
                    ↓
        Amazon EMR Cluster (Apache Spark)
                    ↓
        Distributed Feature Engineering (PySpark)
          - Lag features, rolling statistics
          - Weather variables, calendar encodings
          - Sequence generation (12-hour windows)
                    ↓
        ML Model Training & Benchmarking
          - Linear Regression / Ridge / Lasso
          - Random Forest / Gradient Boosting
          - XGBoost / LightGBM / H2O AutoML
                    ↓
        Evaluation (RMSE, MAE, R²) + Visual Reports
```

---

## Dataset

The dataset consists of large-scale regional electricity consumption records collected at hourly intervals across five U.S. PJM grid regions. All data was stored in S3 in partitioned Parquet format for optimized Spark I/O.

**Key characteristics:**
- Over **900,000+ hourly records** across all regions
- Partitioned as: `region → year → month → parquet files`
- Each region contains approximately **891,496 rows** after feature engineering (DAYTON: 178,300)

**Schema fields:**

| Field | Description |
|---|---|
| `timestamp` | Hourly datetime index |
| `load_mw` | Electricity load in megawatts (target) |
| `temperature` | Ambient temperature |
| `humidity` | Relative humidity |
| `wind_speed` | Wind speed |
| `lag_1h` | Load 1 hour prior |
| `lag_24h` | Load 24 hours prior |
| `lag_7d` | Load 7 days prior |
| `rolling_mean_24h` | 24-hour rolling average |
| `is_holiday` | Binary holiday indicator |
| `is_weekend` | Binary weekend indicator |
| `temp_x_weekend` | Interaction: temperature × weekend |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Distributed Computing | Apache Spark (PySpark / Spark MLlib) |
| Cloud Storage | Amazon S3 |
| Cloud Compute | Amazon EMR (Hadoop cluster) |
| ML Models | XGBoost, LightGBM, Random Forest, Gradient Boosting, Ridge, Lasso, Linear Regression |
| AutoML | H2O AutoML |
| Language | Python 3.9 |
| Data Format | Parquet |
| Visualization | Matplotlib, Seaborn |
| CLI Tools | AWS CLI, SSH, SCP |

---

## Pipeline Stages

### 1. Data Ingestion
Raw hourly electricity consumption Parquet files are loaded from S3 into distributed Spark DataFrames, partitioned by region to maximize parallelism across the EMR cluster.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LoadFeatures").getOrCreate()
df = spark.read.parquet("s3://bigdata-electricity-features-meghanarabba/region=AEP/")
df.printSchema()
```

### 2. Preprocessing
- Corrected data types for all numeric fields (load, temperature, humidity, wind speed, lag features)
- Extracted temporal attributes (year, month, hour, day of week)
- Forward and backward fill for missing target values
- Weather field interpolation to maintain realistic time transitions
- StringIndexer encoding for categorical region field

### 3. Feature Engineering
A uniform feature engineering pipeline was applied across all five regions to ensure consistency:

- **Lag features:** t-1h, t-24h, t-7d — capturing short and long-term consumption history
- **Rolling statistics:** mean, std, min, max over 3, 6, 12, and 24-hour windows
- **Weather variables:** temperature, humidity, wind speed
- **Calendar indicators:** hour of day, day of week, month, holiday flags, weekend flags, season
- **Cyclical encodings:** sin/cos transforms of hour and day-of-week to capture repeating daily and weekly cycles
- **Interaction terms:** temperature × weekend, temperature × holiday

### 4. Sequence Creation
- A **12-hour sliding window** was used for supervised sequence generation: the previous 12 hours predict the next hour
- Each sequence contains 20 engineered features per timestep, resulting in a **240-dimensional input vector**
- This allows classical ML models to leverage time-dependent structure without being sequential architectures

### 5. Model Training
Seven supervised machine learning models were trained and evaluated per region using an **80/20 train-test split**:

1. Linear Regression — baseline linear model
2. Ridge Regression — regularized, addresses multicollinearity
3. Lasso Regression — sparse model with implicit feature selection
4. Random Forest Regressor — ensemble capturing nonlinearity and interactions
5. Gradient Boosting Regressor — sequential ensemble refining errors stage by stage
6. XGBoost — optimized gradient boosting; consistently top performer
7. LightGBM — leaf-optimized boosting, fast on large datasets
8. H2O AutoML — automated model search for comparative benchmarking

### 6. Evaluation
Models are assessed using three metrics:

- **RMSE** (Root Mean Square Error) — penalizes large deviations; critical for grid planning
- **MAE** (Mean Absolute Error) — average expected error in MW; intuitive for real-time use
- **R²** (Coefficient of Determination) — proportion of variance explained; closer to 1.0 is better

---

## Results

### Region AEP

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **XGBoost** | **262.89** | **160.03** | **0.9889** |
| LightGBM | 291.88 | 202.00 | 0.9862 |
| Ridge | 292.88 | 171.59 | 0.9861 |
| Gradient Boosting | 317.15 | 230.67 | 0.9838 |
| Random Forest | 333.75 | 175.72 | 0.9820 |
| Lasso | 333.97 | 211.43 | 0.9820 |
| Linear Regression | 392.97 | 179.06 | 0.9751 |

### Region COMED

| Model | RMSE | R² |
|---|---|---|
| **XGBoost** | **184.86** | **0.993** |
| LightGBM | 245.33 | 0.9878 |
| Random Forest | 201.47 | 0.9917 |

### Region DOM

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **XGBoost** | **212.53** | **130.82** | **0.9926** |
| Random Forest | 221.96 | 129.63 | 0.9920 |
| Linear Regression | 306.99 | 157.98 | 0.9846 |

### Region PJM

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest** | **0.000** | **0.000** | **1.000** |
| XGBoost | 1.936 | 1.936 | 0.000* |
| Gradient Boosting | 5.167 | 3.422 | 0.000* |
| Linear Regression | 79.093 | 54.783 | 0.000 |

*Linear models completely failed for PJM, indicating strong nonlinearity and multicollinearity beyond linear representability.

### Region DAYTON

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **XGBoost** | **34.06** | **21.92** | **0.9918** |
| Random Forest | 36.83 | 22.97 | 0.9905 |
| Linear Regression | 61.79 | 25.98 | 0.9731 |

DAYTON exhibited the lowest absolute error levels across all regions, confirming it as the most predictable load profile in the dataset.

---

## Key Findings

**XGBoost is the strongest overall model**, reaching R² near 0.99 across most regions and consistently achieving the lowest RMSE and MAE. Its ability to model nonlinear temporal interactions and complex weather-load dependencies makes it the optimal choice for production forecasting.

**Lag features dominate predictive power.** Correlation analysis across all regions confirmed that lag_1h, lag_24h, lag_7d, and the 24-hour rolling mean are the strongest predictors of electricity load. This is consistent with real-world consumption behavior driven by historical patterns.

**Regional behavior is distinct.** AEP and COMED exhibit smoother, more predictable trends. DAYTON shows smaller but more volatile swings. DOM and PJM display stronger seasonal variability. Region-specific tuning is necessary for production-grade deployment.

**Linear models are insufficient.** Across all regions, linear approaches captured general trends but consistently failed to model nonlinear seasonal cycles, weather-driven spikes, and lag-based temporal dependencies. Ensemble methods are non-negotiable for this problem.

**Big Data infrastructure is essential.** Processing 900,000+ records with rolling windows, sequence generation, and multi-model training on a local machine was infeasible. Spark on EMR enabled parallel distributed computation that scaled efficiently with data volume.

---

## Challenges Faced

- **Multi-region data scale** — Rolling averages, lag generation, and sequence creation across five regions created significant computational overhead, necessitating the shift from local execution to EMR.
- **S3 configuration** — IAM permission issues, AccessDenied errors, and partial uploads required careful bucket configuration and verification before pipeline execution.
- **Dependency compatibility** — The EMR environment required matching versions of PySpark, Python libraries, and Java; mismatches initially caused runtime failures.
- **Training cost and time** — XGBoost, Random Forest, and Gradient Boosting on large 240-dimensional sequence datasets required careful balancing of accuracy against compute cost on EMR nodes.
- **Output management** — EDA plots, prediction charts, and model files were generated per region; consistent path management was essential during long-running distributed jobs.

---

## Getting Started

**Prerequisites**
- AWS account with EMR and S3 access configured
- Python 3.9+
- PySpark 3.x
- H2O.ai (for AutoML benchmarking)

**Setup**

```bash
# Clone the repository
git clone https://github.com/RohanSinghR/distributed-electricity-forecasting

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure

# Upload data to S3
aws s3 cp data/ s3://your-bucket/electricity-data/ --recursive

# Launch EMR cluster and submit pipeline
bash scripts/launch_emr.sh
```

**Loading data with PySpark on EMR**

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ElectricityForecasting").getOrCreate()
df = spark.read.parquet("s3://your-bucket/region=AEP/")
df.printSchema()
```

---

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── ingest.py          # S3 data loading
│   ├── features.py        # Feature engineering pipeline
│   ├── train.py           # Model training across regions
│   └── evaluate.py        # Metrics and visualization
├── scripts/
│   └── launch_emr.sh      # EMR cluster bootstrap
├── output/
│   ├── images/            # EDA and prediction plots per region
│   └── models/            # Saved model files per region
├── requirements.txt
└── README.md
```

---

## Acknowledgments

- **Prof. Joseph Rosen** — for guidance and feedback throughout the project
- **Illinois Institute of Technology** — for providing resources and infrastructure support

---

## License

MIT License. See `LICENSE` for details.

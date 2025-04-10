# Car Accident Prediction for Fleet Management for FussionSite 

## I. Background and motivation: 
FusionSites Services is a waste management service that relies on a large fleet of specialized vehicles for efficient waste collection and disposal. Ensuring fleet safety and on-time operations is paramount to maintaining regulatory compliance and averting reputational risks.

This project addresses the critical goal of predicting and preventing vehicle accidents. By integrating various data sources—driver behavior, state crash reports, and weather data—FusionSites aims to create a predictive framework that identifies high-risk locations and times for potential accidents, thereby optimizing fleet routing, driver safety, and overall service reliability.

This project goal is to predict car accident using binary classification using drivers' driving patterns, their behaviors, state crash data and weather data.  

## Notebook Description Overview 
### A. Motive EDA and Data Processing notebooks
- **motive_basic_eda.ipynb**
   - Purpose: This conduct very basic exploratory data analysis (like statistic summary, variables distributions) on Motive data on the raw data itself and doesn't do complex calculations to answer specific EDA questions. The goal is to understand the nature of the data and what kind of information can the data provide to use 
  - Outcome: Provides insights into table definitions, column descriptions, and initial data quality check
  - 
- **motive_data_preprocessing.ipynb**  
  - Purpose: This notebook is used as our preliminary processing test to see how can we joined all the Motive tables (inspections, driving periods, combined events, idle events) into a unified data format.  

- **motive_eda_feature_selection.ipynb**  
  - Purpose: This notebook uses for slightly complex EDA questions and create Motive's data features engineer   
  - Outcome: 1 table that can be joined with external data with all the features engineer components created 
 
#### B. External data processing notebooks
- **state_crash_data_processing.ipynb**  
  - Purpose: Preprocess and explore FMCSA state crash data.  
  - Tasks:  
    - Aggregate monthly county-level crash counts, fatalities, injuries, and involved vehicles.  
    - Perform data cleaning, outlier detection, and formatting.  
  - Outcome: Generates a curated dataset of crash statistics by state, county, month, and year.

- **accident_data_location_processing.ipynb**  
  - Purpose: Refine accident records to extract and standardize relevant location information (mapping accident coordinates to counties or ZIP codes).  
  - Outcome: Ensures geographic accuracy, enabling robust spatial joins with other external data sources.

- **weather_data_processing.ipynb**  
  - Purpose: Preprocess daily precipitation data from PRISM, integrating it with accident and driver datasets.  
  - Outcome: Aggregates or interpolates weather variables for the same geographic and temporal resolution used in accident predictions.

- **mapping_external_location_data_to_accidents.ipynb**  
    - Purpose: Combine accident data with external datasets (PRISM precipitation, ZIP code shapefiles, state crash data, insurance claims) based on time and location.  
    - Outcome: Creates a single comprehensive dataset containing accident occurrences, driver behaviors, and environmental factors for each region.
  
  -----------

Extras notebook here: 
  -----------

### C. Joined External and Internal Data
**data_aggregate_nb.ipynb**
- Purpose: this notebook is used to combining external and internal data and further transform categorical type into One Hot Encoding. Afterward, we would split the data into train and test with threshold of 2024-11-01. Train data would be all the driver's trip before the threshold date. We are trying to predict after the threshold, which is our test data
- Outcome: Train and Test data with columns that are ready for model training
  
### D. Model Training and Model Evaluation 
**gbt_hypertraining.ipynb** and **rf_hypertuning.ipynb**
- Purpose: These notebooks train classification models using Gradient-Boosted Trees (GBT) and Random Forest (RF). They apply stratified cross-validation to ensure balanced representation of accident events in each fold. After identifying the best hyperparameters, the final model is evaluated on the held-out test set.  
- Outcome: Tuned models with evaluation results based on test performance.

## II. Data Overview

<p align="center">
  <img src="data_source.jpg" alt="Data Source Overview" width="600"/>
</p>

This project leverages multiple data sources to provide a comprehensive view of driving conditions, driver behavior, and environmental factors. Below is a concise summary of each dataset:
A. Motive Data
    - Purpose: Track driver behavior and vehicle conditions (hazard event types, driving periods, idle events, car inspections).
    - Usage: Correlate risky driving behaviors with crash events and generate internal risk metrics for drivers and vehicles.
B. Insurance Claims Data
    - Description: Contains records of insurance claims including the date, time, location, and driver information for accidents involving FusionSites’ fleet.
    - Usage:
        - Identify and confirm actual accidents among all recorded trips.
        - Map claims back to Motive data to pinpoint which specific driving periods or events correspond to confirmed accidents.
C. State Crash Data (FMCSA’s MCMIS)
    - Source: Federal Motor Carrier Safety Administration (FMCSA).
    - Description: Contains crash records (fatalities, injuries, vehicles involved) for trucks and buses reported by states.
    - Time Range: 2023–2025 (project scope).
    - Usage: Compute monthly, county-level crash statistics (e.g., number of crashes, fatalities, injuries). These aggregated statistics serve as ground truth for accident frequency trends.
D. Precipitation Data (PRISM)
    - Source: PRISM Climate Group at Oregon State University.
    - Time Range: Daily precipitation data from January 1, 2023 to January 31, 2025.
    - Spatial Resolution: 4km.
    - Usage: Incorporate weather context (especially precipitation) into accident risk modeling.
    - Note: Data is “stable,” “provisional,” or “early results” depending on recency.

F. ZIP Code & Geographic Data
    - 2020 Census 5-Digit ZIP Code Tabulation Area (ZCTA5) Shapefiles
        - Source: U.S. Census Bureau TIGER/Line series, updated January 27, 2024.
        - Usage: Match accident records and precipitation data to ZIP codes and counties for location-based analyses.
    - US Cities/States Data
        - Usage: Geospatial referencing and consistent location naming across datasets.

## III. Tool& Infrastructure
### Core Technologies 
- Azure Databricks: Used for interactive analytics and distributed data processing on the Azure platform.
- Google Cloud Dataproc: Provides a managed Hadoop and Spark cluster environment for large-scale data processing.
- HDFS (Hadoop Distributed File System): Facilitates storage of large datasets in a distributed manner, enabling efficient parallel operations.
- Spark: Employed for data parallelization, feature engineering, and large-scale model training.
### General WorkFlow: 
1. Data storage in HDFS  
   After data cleaning and aggregation, we store both internal (FusionSites data) and external data in HDFS. While this increases storage needs, it prevents re-running heavy computations in subsequent steps.
2. Feature engineering in a distributed environment  
   We merge cleaned datasets in HDFS and use Spark on Databricks or Dataproc for parallel feature engineering. This approach handles large data volumes efficiently and shortens processing times.
3. Model training with Spark  
   We create train and test sets in HDFS and leverage Spark’s distributed ML libraries. Databricks or Dataproc resources scale on demand, keeping training both efficient and cost-effective.
4. Python for final model execution  
   Final model training and tuning run in Python scripts to reduce notebook overhead. This approach simplifies automation and keeps iterative experiments efficient.
By saving intermediate outputs after each step, we avoid repetitive heavy-lift transformations, preserving data lineage and supporting iterative experimentation in a production-scale environment.

## IV. Methodology
### 1. Data Processing 

<p align="center">
  <img src="external_data_processing.jpg" alt="External Data Processing" width="45%" style="margin-right: 10px;"/>
  <img src="internal_data_processing.jpg" alt="Internal Data Processing" width="45%"/>
</p>

<p align="center">
  <em>We process and engineer features from external and internal data sources separately. External data includes crash statistics, precipitation, and geographic mappings, while internal data includes Motive driver activity and inspection logs. These are then joined by location (ZIP/county) and time (date/month) to create a unified, feature-rich dataset used for modeling.</em>
</p>

To build a comprehensive dataset for accident prediction, we processed and joined both internal FusionSites data and external datasets, such as crash statistics and weather information. Our approach focused on cleaning, standardizing, and aligning records across systems to ensure consistent geographic and temporal matching.

-  Internal data cleaning (FusionSites): We cleaned and prepared four main tables related to driver behavior: hazard events, driving periods, inspections, and idle events. This involved removing nulls, filtering outliers, renaming columns, and extracting trip-level date information. These datasets were then joined by `driver_id`, `vehicle_id`, `trip_date`, and `main_event_type` to create a master dataset that captures all activity for each trip.

- External data integration: We incorporated accident records, insurance claims, and environmental data into the trip dataset. For accident records, we matched location fields like ZIP code, city, and state using fuzzy matching and manual corrections. Precipitation data was extracted from spatial raster files and mapped to ZIP codes. Both types of data were joined to trip records using date and location keys.

- Geographic and temporal alignment: We standardized ZIP codes, state and county identifiers, and handled location ambiguities using regional FusionSites data. For each accident, we added county-level crash statistics and average precipitation data. Where direct matches were unavailable, we implemented fallback strategies using nearby ZIP codes or adjacent time windows.

- Data aggregation and feature readiness: We grouped data at the monthly county level and computed key metrics such as crash counts, fatalities, injuries, and vehicles involved. Precipitation data was reshaped and categorized to support feature engineering. Special care was taken to avoid data leakage by ensuring only historical data is used in modeling.

### 2. Features Engineering 
Feature engineering is conducted separately on internal and external datasets to ensure that all variables are clean, structured, and informative before joining them into a unified dataset. The result is a single, feature-rich dataset that links driving behavior, environmental factors, and crash outcomes at a daily and county level. This dataset is cached for efficient exploration and modeling in downstream notebooks.

#### External data feature engineering

**State crash data (`yearly_crash_data_processing.ipynb`)**  
- No missing or non-numerical variables were used.  
- Raw crash-level data was grouped by crash year, month, state, and county.  
- Aggregated features include:  
  - Total number of crashes  
  - Total fatalities  
  - Total injuries  
  - Total vehicles involved  
- Output: `state_crash_monthly_county_counts`

**Precipitation data**  
- Used daily total precipitation by ZIP code from 2023 to 2025.  
- Transformed from wide to long format.  
- Aggregated at the site-date level to compute:  
  - Mean, median, std dev, quartiles, min, max, and IQR.  
- Output: `final_precipitation_stats`

**ZIP code shapefile and FusionSites data (`site_radius_by_zipcode.ipynb`)**  
- Transformed geographic coordinates to appropriate CRS for distance calculations.  
- Calculated distances between each FusionSites ZIP code and surrounding ZIP codes (within a 40-mile radius).  
- Derived features:  
  - Target ZIP codes per site  
  - Distance to each ZIP  
- Output: Intermediate site-ZIP distance tables

**Crash + Precipitation mapping to sites (`map_external_data_to_sites.ipynb`)**  
- Mapped aggregated crash and weather data to site locations via ZIP and county codes.  
- Applied inner joins with distinct filters to avoid duplication.  
- Output:  
  - `aggregated_site_radius_crash_df`  
  - `final_precipitation_stats`

**Location data feature engineering (`location_data_feature_engineering.ipynb`)**  
- Aggregated crash stats by site, brand, year, and month.  
- Created rolling-window statistics (mean, median, std, IQR, etc.) using Spark window functions.  
- Calculated moving averages for crash metrics (1–6 months) and precipitation metrics (1–3 days).  
- Generated a complete calendar with all site-date combinations to ensure temporal coverage.  
- Output: Final engineered CSV file ready to be merged with Motive data.

#### Internal data feature engineering (Motive + driver-level data)

The following tables are cleaned, transformed, and joined by `driver_id`, `vehicle_id`, and `trip_date`:

**Drivers Trips Information**
- Includes trip frequency, average speed, distance, recency, and rolling trip stats:
  - `trip_date_distance`, `trip_date_minutes`, `trip_date_avg_speed_mph`
  - Rolling features like `rolling_7trip_avg_speed_mph`, `rolling_30day_total_distance`
  - Trip behavior change metrics: `change_in_distance`, `change_in_minutes`

**Hazard Event Information for each trip**
- Event-based features for each trip:
  - Total and type-specific event counts (e.g., `speeding`, `drowsiness`, `crash`, roughly 15+ different hazard events from Motive)
  - Rolling 7/15/30-day sums per event type
  - Speeding severity breakdowns (`low`, `mid`, `high`) and their respective rolling stats
  - Ratios per distance, minutes, and events to normalize behavior
  - Coaching and review metrics (e.g., `prev_review_rate_per_km`)

**Vehicle Inspection per each trip date**
- Inspection-based features for each vehicle-trip:
  - Cumulative inspection counts and issue rates
  - Days since last inspection
  - Rolling inspection/activity ratios across previous 7/15/30 trips

** Driver Idling Events **
- Idle behavior for each trip:
  - `idle_event_count_per_trip`, `avg_idle_duration_per_trip`, `total_idle_minutes_per_trip`
  - Rolling 7, 15, 30 -trip averages for idle count, duration, and minutes

** Trips and corresponding sites **
This is used to understand where would this trip dispatched from, allowing to mapped with weather data within that region
- Geographic link between trips and site location:
  - `zipcode`, `motive_group_id` for mapping driver behavior to site-level risk

These table are going to be joined together with following shared keys `driver_id`, `vehicle_id`, `trip_date`, and `group_id`.

### 3. Final aggregation & train-test split

The final modeling dataset is created by merging internal and external engineered features using shared keys such as `driver_id` and `trip_date`. The unified feature table contains:

- Behavior metrics at the trip level  
- Rolling trends across time and trip history  
- Inspection and idle summaries  
- Demographic and tenure data  
- Location-aware features (ZIP/site linkage)  
- External crash and precipitation statistics at the site-date level  

This final dataset is saved in Parquet format to enable efficient reuse for modeling and evaluation workflows.

#### Data preparation for modeling

Before splitting the data, we perform the following steps:
- Remove leakage-prone columns: Any features that leak future information (e.g., cumulative trip counts as of the current trip) are excluded.
- Handle categorical and date variables**: Non-numerical fields such as `trip_date` are either removed or converted using one-hot encoding.
- Address class imbalance: The dataset contains a significant imbalance (~368 accident trips vs. 201,498 non-accident trips). To reduce bias while preserving structure, we apply undersampling by removing all non-accident trips that occurred before the first recorded accident trip. This helps maintain data integrity while improving class balance.

#### Train-test split

We split the data based on a fixed temporal threshold:
- Training set: All trips before `2024-11-01`  
- Test set**: All trips on and after `2024-11-01`  
The split above will ensure a rough 80-20 split between train and test sets.

### 4. Model Training 
We trained two classifiers: **Random Forest (RF)** and **Gradient Boosted Trees (GBT)**, each with their own tuning and evaluation pipelines.

#### Random Forest Classifier

- We manually performed 4-fold cross-validation for hyperparameter tuning to address class imbalance.
- Tuned parameters included:  
  - `numTrees`: [50, 60, ..., 120]  
  - `maxDepth`: [5, 10, 15]  
  - `maxBins`: [32, 64]  
- After selecting the best configuration (110 trees, depth of 5, 64 bins), we retrained the model on the full training set.
- Feature importance analysis revealed 26 features with zero impact; these were dropped before final training.
- The model was retrained after filtering and evaluated on a holdout set for unbiased performance assessment.

#### Gradient Boosted Trees Classifier

- Hyperparameter tuning was performed using **Optuna** with a Tree-structured Parzen Estimator (TPE).
- Initial tuning explored:
  - `max_depth`: [3, 15]  
  - `step_size`: [0.05, 3]  
  - `subsample_rate`: [0.5, 1.0]  
  - `min_instances_per_node`: [1, 20]  
- After selecting the top 5 models via stratified validation, we performed feature importance filtering, dropping 85 non-informative features.
- A second Optuna round refined `step_size`, `subsample_rate`, and `min_instances_per_node`, with `max_depth` fixed at 3.
- Final model was trained using optimal hyperparameters on the full training data.

### 5. Model Performance on Test Data
#### Random Forest

- **AUC**: 0.5452 (barely above random)  
- **AUPRC**: 0.00276 (very low)  
- **Precision** (overall): 0.9953  
- **Recall** (overall): 0.8716  
- **True Positive Rate (accidents only)**: 16.67%  
- **Positive Predictive Value (accidents only)**: 0.31%

> Despite strong overall precision and recall (due to class imbalance), the model struggles to identify true accidents. Only 0.31% of flagged accidents are correct.

#### Gradient Boosted Trees

- **AUC**: 0.9110  
- **AUPRC**: 0.0126  
- **Precision** (overall): 0.9982  
- **Recall** (overall): 0.8026  
- **F1 Score**: 0.8890  
- **False Positive Rate**: 19.75%  
- **Positive Predictive Value (accidents only)**: 0.68%

> The GBT model achieves a high AUC but suffers from an extremely low positive predictive value. Only 0.68% of predicted accidents are correct, indicating many false alarms.

### 6. Key takeaways

- Both models surfaced meaningful features, including prior accidents, safety events, inspection history, and driving tenure.
- GBT outperformed RF in terms of AUC but still lacked precision on accident prediction due to data imbalance.
- The quality of the prediction is ultimately limited by data quality: missing values, inconsistent adoption of Motive, and unreliable accident records.
- Future modeling should focus on improving data coverage, especially around route information, accident validation, and consistent sensor usage.
- Several key features were consistently important across both the Random Forest and Gradient Boosted Tree models:
  - `vehicle_cum_issues`: History of vehicle issues is a strong predictor of accident risk.
  - `rolling_15trip_avg_speed_mph`: Consistent high speeds over recent trips are associated with higher risk.
  - `rolling_15day_total_distance`: Recent driving intensity plays a significant role in crash likelihood.
  - `prev_trip_date_distance`: Distance from the previous trip may indicate driver fatigue or exposure.
 _ Additional insights from feature importance analysis
  - Long-term driving patterns are more predictive than single-trip behavior.
  - GBT emphasized distraction-related events (e.g., cell phone use, drowsiness), while RF relied more on historical ratios and accident history.
  - Speed-related features were important when considered over time, rather than as isolated events.
  - Vehicle condition and maintenance history should be monitored closely for risk management.
  
> A more practical short-term solution could be deploying a **dashboard** highlighting high-risk behaviors (e.g., speeding, harsh braking, overdue inspections) rather than deploying a low-confidence predictive model.

## 5. Challenges and Project Limitations: 


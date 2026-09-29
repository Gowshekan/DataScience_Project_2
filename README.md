# DS_Day01_19: Restaurant Staffing Analysis

## Data provenance

The workspace did not contain a source dataset when this project was started. At the user's direction, `data/restaurant_staffing.csv` is a reproducible, fixed-seed **synthetic teaching dataset**, not actual observations from a restaurant chain. Its generated demand patterns, staffing, preparation times, and queues are illustrative only. Do not use its findings as operational evidence. Replace the CSV with an authorized real dataset to analyze real outlets; the analysis script reads that file without overwriting it.

## Problem statement and objectives

Explore how hourly demand, staffing, preparation time, and queue pressure vary across outlets, weekdays, weekends, and flagged holidays. The analysis identifies observed peak periods, calculates orders per employee, applies a transparent data-derived staffing classification, compares outlets, and evaluates one Random Forest model for queue-length estimation. Associations are descriptive and do not establish causation.

## Dataset columns

`Date`, `Outlet_ID`, `Hour`, `Order_Volume`, `Average_Preparation_Time`, `Queue_Length`, `Staff_Count`, `Day_Type`, and `Holiday`.

## Methods

- Cleaning parses dates and numeric fields, removes exact duplicates, drops rows missing a valid date/outlet/hour, replaces invalid negative measures with missing values, and imputes numeric gaps with outlet-hour medians followed by the overall median. Invalid day types are derived from the date; unrecognized holiday flags are treated as non-holiday. Counts are rounded to whole numbers. Unusual high values are retained.
- Descriptive statistics include count, mean, median, standard deviation, minimum, and maximum for order volume, preparation time, queue length, and staff count.
- EDA and figures cover hourly demand, preparation time, queue length, staff count, day type, staffing versus queue length, orders per employee, and outlet comparisons.
- Peak periods are groups at or above the 90th percentile of group-average order volume for each requested grouping (hour, outlet-hour, day-type-hour, and holiday-hour).
- Orders per employee is `Order_Volume / Staff_Count`; zero staff is represented as undefined rather than dividing by zero.
- Staffing labels use dataset-derived quartiles: understaffed if workload exceeds its 75th percentile, or queue exceeds its 75th percentile while preparation time exceeds its median; overstaffed if workload is below its 25th percentile, queue is at or below its median, and preparation time is below its median; otherwise balanced. The inclusive queue threshold also works when the observed queue median is zero. The labels are heuristic indicators, not verified staffing requirements.
- Queue length is treated as an available queue-pressure indicator, not direct waiting time. Correlations and grouped statistics are observational; they cannot show that staffing causes a change in queues.
- The sole ML algorithm is a Random Forest regressor predicting `Queue_Length`. Inputs are hour, outlet, order volume, average preparation time, staff count, day type, and holiday. The target is excluded from inputs. Categorical fields use one-hot encoding. A chronological 80/20 split by unique date reduces future-to-past leakage. MAE, MSE, RMSE, R-squared, and feature importances are reported; importance is not causation.

## Install and run

From this project directory, install the listed libraries and run:

```powershell
python -m pip install -r requirements.txt
python run_analysis.py
```

Run `notebooks/restaurant_staffing_analysis.ipynb` in Jupyter, or execute it from the project directory with:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/restaurant_staffing_analysis.ipynb
```

If `data/restaurant_staffing.csv` is absent, `run_analysis.py` creates it using a fixed random seed. If the CSV already exists, it is loaded and not overwritten.

## Generated outputs

The run creates a cleaned CSV, descriptive statistics, peak-period table, staffing classifications, outlet comparison, model evaluation and feature importance, queue relationship tables, practical action plan, six calculated insights, and PNG figures. The model is saved as `models/random_forest_model.pkl`.

## Project structure

```text
DS_Day01_19_Restaurant_Staffing/
├── data/restaurant_staffing.csv
├── notebooks/restaurant_staffing_analysis.ipynb
├── outputs/
│   ├── figures/
│   ├── cleaned_dataset.csv
│   ├── descriptive_statistics.csv
│   ├── peak_demand_periods.csv
│   ├── staffing_analysis.csv
│   ├── outlet_comparison.csv
│   ├── model_evaluation.csv
│   ├── feature_importance.csv
│   ├── staff_queue_correlations.csv
│   ├── staff_queue_grouped_statistics.csv
│   ├── practical_action_plan.csv
│   └── key_insights.txt
├── models/random_forest_model.pkl
├── requirements.txt
├── README.md
└── run_analysis.py
```

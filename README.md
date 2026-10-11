# Agricultural-Land-use-Pattern-Analysis-Tool

## 1. Project Overview

The **Agricultural Land-Use Pattern Analysis Tool** is a Python-based project that analyzes historical agricultural land-use patterns across Indian states and districts.

The project uses historical land-use data to explore changes over time, compare geographical regions, examine cropping patterns, and visualize differences among land-use categories.

The tool uses data preprocessing and Exploratory Data Analysis (EDA) to convert raw agricultural data into meaningful visual representations. Statistical analysis and forecasting are additional components of the project.

## 2. Problem Statement

Agricultural land is an important resource for India's agricultural sector and food production. Land is classified into different categories, including net area sown, cropped area, forests, fallow land, and other land-use classifications. These patterns may change over time and vary across states and districts.

The objective of this project is to analyze historical agricultural land-use patterns across Indian states and districts using data preprocessing, visualization, statistical analysis, and forecasting techniques.

The project aims to identify trends over time, compare geographical differences, examine cropping patterns, and understand the patterns present in the available historical data.

## 3. Objectives

- Analyze historical agricultural land-use patterns across Indian states and districts.
- Study changes in land-use metrics over time.
- Compare land-use patterns across geographical regions.
- Examine cropping intensity and land-use composition.
- Visualize trends, distributions, relationships, and missing values.
- Prepare cleaned datasets for further analysis.
- Apply statistical techniques to quantify trends and changes.
- Explore forecasting methods for estimating possible future trends.

## 4. Dataset

**Dataset name:** `classification-of-area.csv`

**Source:** India Data Portal — Land Use Statistics

Dataset link: https://ckandev.indiadataportal.com/dataset/land-use-statistics

The data includes historical agricultural land-use records with information about years, states, districts, land-use metrics, classifications, and area values.

### Important columns

| Column | Description |
|---|---|
| `year` | Agricultural year recorded in the source dataset |
| `state_name` | Name of the state or Union Territory |
| `state_code` | Geographic code for the state, where available |
| `district_name` | Name of the district |
| `district_code` | Geographic code for the district, where available |
| `land_use_metrics` | Main land-use measurement |
| `area_classification` | Broad land-use classification |
| `area_sub_classification` | More specific land-use classification |
| `area` | Reported area value |

The preprocessing module also creates additional columns, including `category` and `start_year`, to support analysis and visualization.

### Important land-use metrics

- Reporting Area
- Net Area Sown
- Cropped Area
- Area Sown More Than Once

Other classifications include forests, fallow land, and land not available for cultivation.

## 5. Technologies Used

- **Python** — Main programming language
- **Pandas** — Data loading, cleaning, grouping, and manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **SciPy** — Statistical calculations and tests
- **Statsmodels** — Time-series analysis and forecasting

## 6. Project Workflow

1. **Data collection:** Load historical agricultural land-use records from the CSV dataset.
2. **Data preprocessing:** Standardize column names, clean text values, handle missing and invalid values, convert data types, and extract the starting year.
3. **Dataset preparation:** Prepare cleaned records for EDA and geographical analysis, along with annual metric totals for statistical analysis.
4. **Exploratory Data Analysis:** Generate visualizations to explore temporal trends, land-use composition, cropping patterns, geographical differences, distributions, and missing values.
5. **Statistical analysis:** Analyze descriptive statistics, trends, yearly changes, and relationships between variables.
6. **Forecasting:** Evaluate forecasting methods using historical annual data and assess their performance.

## 7. Project Structure

```text
Agricultural-Land-use-Pattern-Analysis-Tool/
│
├── app.py
├── terminal_app.py
├── data_preprocessor.py
├── eda_visualizer.py
├── stat_forecasting.py
├── classification-of-area.csv
├── README.md
└── eda_outputs/
```

**Note:** `app.py` is the application interface scaffold. `terminal_app.py` provides the console-based application menu. The exact files available may vary as development progresses. The `eda_outputs/` directory is generated when visualizations are saved.

## 8. Module Description

### `data_preprocessor.py`

Responsible for loading and preparing the dataset.

Main functions:

- `preprocess_data()` — Cleans and standardizes the input DataFrame.
- `create_member2_dataset()` — Provides cleaned records for visualization and geographical analysis.
- `create_member3_dataset()` — Creates annual totals for each land-use metric and calculates absolute and percentage changes.

### `eda_visualizer.py`

Responsible for Exploratory Data Analysis and visualization.

It provides interactive options to generate charts, compare trends, inspect land-use patterns, and examine data quality.

### `stat_forecasting.py`

Intended to provide statistical analysis and forecasting functionality, including descriptive statistics, trend analysis, yearly changes, correlation analysis, model evaluation, and annual forecasts.

### `terminal_app.py`

Provides the console-based main menu and connects the preprocessing, visualization, and other analysis modules.

### `app.py`

Contains the application interface scaffold.

## 9. Installation and Requirements

### Prerequisites

- Python 3.10 or later recommended
- pip
- Git (optional, for cloning the repository)

### Install dependencies

Open a terminal in the project directory and run:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

If the interface in `app.py` uses additional packages, install those packages as well.

## 10. How to Run

### Run the console application

```bash
python terminal_app.py
```

The console application loads the prepared datasets and displays the main menu.

### Run the EDA module independently

```bash
python eda_visualizer.py
```

The module loads the source dataset through the preprocessing functions and opens the EDA menu.

## 11. Team Contributions

The project is organized into separate modules to support collaborative development:

- **Data preprocessing:** Loading, cleaning, and preparing datasets.
- **EDA and visualization:** Exploring land-use patterns through charts and summaries.
- **Statistical analysis and forecasting:** Quantifying trends and evaluating forecasting methods.
- **Application integration:** Connecting the modules through the application interface and menus.

Individual contributions can be specified here according to the team's final agreed responsibilities.

## 12. Conclusion

The Agricultural Land-Use Pattern Analysis Tool aims to make historical agricultural land-use data easier to explore and interpret. By combining data preprocessing, visualization, statistical analysis, and forecasting, the project provides a structured approach to investigating temporal and geographical differences in agricultural land-use patterns across India.

The findings can help describe historical patterns and identify areas for further investigation, while recognizing that the analysis does not independently establish the causes of those patterns.

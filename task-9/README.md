# Task 9a: Time-Oriented Data Analysis and Visualization

## Quick Start

### Run the Analysis
```bash
python task_9a_time_series_analysis.py
```

## Output Files

### Visualizations (Images)
1. **9a_line_graphs_trends.png** - 4 line graphs showing call duration trends
2. **9a_area_charts.png** - 2 area charts for distribution visualization
3. **9a_trend_analysis.png** - 4 different trend line methods

### Python Code
- **task_9a_time_series_analysis.py** - Minimized analysis script

## Dataset
- **telecom_customer_call_records_100.csv** - 100 telecom call records

## Analysis Features

### ✓ Line Graphs
- Individual call durations over time
- Daily average with trend line
- Total daily call volume
- Daily call count

### ✓ Area Charts
- Daily total duration (filled area)
- Call duration with 10-period moving average

### ✓ Trend Analysis
- Linear regression trends with R² scoring
- Polynomial trends (2nd degree)
- Multiple moving average trends (7-day, 14-day)
- Cumulative duration trends

## Key Metrics Analyzed

- Call Duration (seconds)
- Average daily duration
- Total daily volume
- Call count per day
- Trend lines and slopes

## Statistical Analysis

- Descriptive statistics (mean, median, std dev)
- Trend analysis (slope, direction)
- R² model scoring
- Moving averages
- Linear and polynomial regression

## Dependencies

```python
pandas
numpy
matplotlib
seaborn
scipy
sklearn
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Results Summary

- **Mean Call Duration**: 1,651.49 seconds (~27.5 minutes)
- **Trend**: Slight increase (+0.89 sec/day)
- **R² Score**: 0.0002
- **Range**: 18 - 3,446 seconds

All visualizations are saved as high-resolution PNG files (300 DPI).

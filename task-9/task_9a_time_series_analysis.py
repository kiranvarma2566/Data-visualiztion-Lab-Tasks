import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load and prepare data
df = pd.read_csv(r'C:\Users\kiran\Downloads\task-9-dv\telecom_customer_call_records_100.csv')
print("=" * 80)
print("TASK 9A: TIME-ORIENTED DATA ANALYSIS\n" + "=" * 80)
print("\nDataset Overview:\n", df.head(10))
print("\nStatistical Summary:\n", df.describe())

# Create time dimension
np.random.seed(42)
df['Timestamp'] = [datetime(2024, 1, 1) + timedelta(hours=i*7.2) for i in range(len(df))]
df['Date'] = df['Timestamp'].dt.date
df.sort_values('Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# Aggregate daily data
daily_stats = df.groupby('Date')['Call_Duration_sec'].agg(['count', 'sum', 'mean', 'std']).reset_index()
daily_stats.columns = ['Date', 'Call_Count', 'Total_Duration', 'Avg_Duration', 'Std_Duration']
daily_stats['Date'] = pd.to_datetime(daily_stats['Date'])

print(f"\nDate Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}\nTotal Records: {len(df)}")

# Prepare common variables
x_numeric = np.arange(len(daily_stats))
X = x_numeric.reshape(-1, 1)
y = daily_stats['Avg_Duration'].values

# Helper function for plot styling
def style_plot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# ============================================================================
# VISUALIZATION 1: LINE GRAPHS AND TRENDS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Time-Oriented Analysis: Call Duration Patterns and Trends', fontsize=16, fontweight='bold', y=0.995)

# Plot configurations
plots = [
    (df['Timestamp'], df['Call_Duration_sec'], '#2E86AB', 'o', 'Timestamp', 'Call Duration (seconds)', 'Line Graph: Individual Call Durations Over Time'),
    (daily_stats['Date'], daily_stats['Avg_Duration'], '#A23B72', 'o', 'Date', 'Average Call Duration (seconds)', 'Daily Average Call Duration with Trend Line'),
    (daily_stats['Date'], daily_stats['Total_Duration']/60, '#06A77D', 's', 'Date', 'Total Call Duration (minutes)', 'Line Graph: Total Daily Call Volume'),
    (daily_stats['Date'], daily_stats['Call_Count'], '#C73E1D', '^', 'Date', 'Number of Calls', 'Line Graph: Daily Call Count')
]

for idx, (x, y_data, color, marker, xlabel, ylabel, title) in enumerate(plots):
    ax = axes.flatten()[idx]
    ax.plot(x, y_data, color=color, linewidth=1.5 if idx==0 else 2, alpha=0.7 if idx==0 else 1, marker=marker, markersize=3 if idx==0 else 5)
    
    # Add trend line for second plot
    if idx == 1:
        z = np.polyfit(x_numeric, daily_stats['Avg_Duration'], 1)
        ax.plot(daily_stats['Date'], np.poly1d(z)(x_numeric), color='#F18F01', linestyle='--', linewidth=2.5, label=f'Trend Line (slope: {z[0]:.2f})')
        ax.legend(loc='best', fontsize=9)
    
    style_plot(ax, xlabel, ylabel, title)

plt.tight_layout()
plt.savefig('9a_line_graphs_trends.png', dpi=300, bbox_inches='tight')
print("\n[SUCCESS] Saved: 9a_line_graphs_trends.png")
plt.show()

# ============================================================================
# VISUALIZATION 2: AREA CHARTS
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Area Charts: Call Duration Distribution Over Time', fontsize=16, fontweight='bold')

# Plot 1: Daily total
axes[0].fill_between(daily_stats['Date'], daily_stats['Total_Duration']/60, color='#5B9BD5', alpha=0.6, label='Total Duration')
axes[0].plot(daily_stats['Date'], daily_stats['Total_Duration']/60, color='#1F4E78', linewidth=2)
style_plot(axes[0], 'Date', 'Total Call Duration (minutes)', 'Area Chart: Daily Total Call Duration')
axes[0].legend(loc='best', fontsize=10)

# Plot 2: With moving average
df['MA_Duration'] = df['Call_Duration_sec'].rolling(window=10).mean()
axes[1].fill_between(df['Timestamp'], 0, df['Call_Duration_sec'], color='#70AD47', alpha=0.5, label='Call Duration Range')
axes[1].plot(df['Timestamp'], df['Call_Duration_sec'], color='#375623', linewidth=1.5)
axes[1].plot(df['Timestamp'], df['MA_Duration'], color='#FF6B35', linewidth=2.5, label='10-Period Moving Average')
style_plot(axes[1], 'Timestamp', 'Call Duration (seconds)', 'Area Chart: Call Duration with Moving Average')
axes[1].legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('9a_area_charts.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved: 9a_area_charts.png")
plt.show()

# ============================================================================
# VISUALIZATION 3: TREND ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Trend Analysis: Identifying Systemic Patterns', fontsize=16, fontweight='bold', y=0.995)

# Linear Regression
model = LinearRegression().fit(X, y)
axes[0, 0].scatter(daily_stats['Date'], y, color='#4A90E2', s=60, alpha=0.6, label='Actual Data')
axes[0, 0].plot(daily_stats['Date'], model.predict(X), color='#E94B3C', linewidth=3, label='Linear Trend')
style_plot(axes[0, 0], 'Date', 'Average Call Duration (seconds)', f'Linear Trend Analysis (R²={model.score(X, y):.3f})')
axes[0, 0].legend(loc='best', fontsize=9)

# Polynomial Trend
p2 = np.poly1d(np.polyfit(x_numeric, y, 2))
axes[0, 1].scatter(daily_stats['Date'], y, color='#50C878', s=60, alpha=0.6, label='Actual Data')
axes[0, 1].plot(daily_stats['Date'], p2(x_numeric), color='#8B4513', linewidth=3, label='Polynomial Trend (deg=2)')
style_plot(axes[0, 1], 'Date', 'Average Call Duration (seconds)', 'Polynomial Trend Analysis (2nd Degree)')
axes[0, 1].legend(loc='best', fontsize=9)

# Moving Averages
axes[1, 0].plot(daily_stats['Date'], y, color='#CCCCCC', linewidth=1, alpha=0.5, label='Original Data')
for window, color in [(7, '#FF6B9D'), (14, '#4ECDC4')]:
    ma = daily_stats['Avg_Duration'].rolling(window=window, min_periods=1).mean()
    axes[1, 0].plot(daily_stats['Date'], ma, color=color, linewidth=2.5, label=f'{window}-Day Moving Average')
style_plot(axes[1, 0], 'Date', 'Average Call Duration (seconds)', 'Moving Average Trends')
axes[1, 0].legend(loc='best', fontsize=9)

# Cumulative Trend
cumulative = daily_stats['Total_Duration'].cumsum() / 3600
axes[1, 1].fill_between(daily_stats['Date'], cumulative, color='#9B59B6', alpha=0.5)
axes[1, 1].plot(daily_stats['Date'], cumulative, color='#2C3E50', linewidth=2.5, label='Cumulative Duration')
style_plot(axes[1, 1], 'Date', 'Cumulative Call Duration (hours)', 'Cumulative Call Duration Trend')
axes[1, 1].legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig('9a_trend_analysis.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Saved: 9a_trend_analysis.png")
plt.show()

# Statistical Summary
stats = df['Call_Duration_sec']
z_slope = np.polyfit(x_numeric, y, 1)[0]
print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS SUMMARY\n" + "=" * 80)
print(f"\n1. TREND ANALYSIS:")
print(f"   - Linear Trend Slope: {z_slope:.4f} seconds/day")
print(f"   - Trend Direction: {'Increasing' if z_slope > 0 else 'Decreasing'}")
print(f"   - R² Score: {model.score(X, y):.4f}")
print(f"\n2. DESCRIPTIVE STATISTICS:")
print(f"   - Mean Call Duration: {stats.mean():.2f} seconds")
print(f"   - Median Call Duration: {stats.median():.2f} seconds")
print(f"   - Std Deviation: {stats.std():.2f} seconds")
print(f"   - Min Duration: {stats.min()} seconds")
print(f"   - Max Duration: {stats.max()} seconds")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!\n" + "=" * 80)
print("\nGenerated Visualizations:")
print("  1. 9a_line_graphs_trends.png - Line graphs showing call duration trends")
print("  2. 9a_area_charts.png - Area charts for distribution visualization")
print("  3. 9a_trend_analysis.png - Multiple trend lines and patterns")
print("\n" + "=" * 80)

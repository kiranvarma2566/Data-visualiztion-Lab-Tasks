import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('agriculture_crop_yield.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Separate categorical and continuous columns
categorical_cols = ['State', 'Crop_Type', 'Season', 'Climate_Zone', 'Soil_Type', 
                   'Irrigation_Type', 'Pest_Infestation_Level', 'Disease_Incidence']

continuous_cols = ['Area_Hectares', 'Yield_Tonnes', 'Yield_per_Hectare', 
                  'Fertilizer_Usage_kg', 'Precipitation_mm', 'Temperature_Celsius',
                  'Storage_Loss_Percentage', 'Market_Price_per_Tonne', 'Total_Revenue']

print(f"\nCategorical columns: {categorical_cols}")
print(f"Continuous columns: {continuous_cols}")

# Create visualizations for categorical data
def create_categorical_visualizations():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Bar Chart - Crop Type Distribution
    crop_counts = df['Crop_Type'].value_counts()
    axes[0, 0].bar(crop_counts.index, crop_counts.values, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Crop Type Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Crop Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Pie Chart - Crop Type Distribution
    axes[0, 1].pie(crop_counts.values, labels=crop_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Crop Type Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # 3. Bar Chart - Season Distribution
    season_counts = df['Season'].value_counts()
    axes[1, 0].bar(season_counts.index, season_counts.values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Season Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Season')
    axes[1, 0].set_ylabel('Count')
    
    # 4. Pie Chart - Season Distribution
    axes[1, 1].pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Season Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create visualizations for continuous data
def create_continuous_visualizations():
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Scatter Plot - Area vs Yield
    axes[0, 0].scatter(df['Area_Hectares'], df['Yield_Tonnes'], alpha=0.6, color='orange')
    axes[0, 0].set_title('Area vs Yield (Scatter Plot)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Area (Hectares)')
    axes[0, 0].set_ylabel('Yield (Tonnes)')
    
    # 2. Line Plot - Temperature by Crop Type
    temp_by_crop = df.groupby('Crop_Type')['Temperature_Celsius'].mean().sort_values(ascending=False)
    axes[0, 1].plot(temp_by_crop.index, temp_by_crop.values, marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_title('Temperature by Crop Type (Line Plot)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Crop Type')
    axes[0, 1].set_ylabel('Average Temperature (°C)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Strip Plot - Yield per Hectare by Crop Type
    sns.stripplot(data=df, x='Crop_Type', y='Yield_per_Hectare', jitter=0.3, alpha=0.6, ax=axes[0, 2])
    axes[0, 2].set_title('Yield per Hectare by Crop Type (Strip Plot)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Crop Type')
    axes[0, 2].set_ylabel('Yield per Hectare (Tonnes)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Swarm Plot - Market Price by Crop Type
    sns.swarmplot(data=df, x='Crop_Type', y='Market_Price_per_Tonne', size=3, ax=axes[1, 0])
    axes[1, 0].set_title('Market Price by Crop Type (Swarm Plot)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Crop Type')
    axes[1, 0].set_ylabel('Market Price per Tonne ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Histogram - Yield per Hectare
    axes[1, 1].hist(df['Yield_per_Hectare'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Yield per Hectare Distribution (Histogram)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Yield per Hectare (Tonnes)')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Density Plot - Temperature
    df['Temperature_Celsius'].plot.kde(ax=axes[1, 2], color='purple', linewidth=2)
    axes[1, 2].set_title('Temperature Distribution (Density Plot)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Temperature (°C)')
    axes[1, 2].set_ylabel('Density')
    
    # 7. Histogram with Rug Plot - Precipitation
    axes[2, 0].hist(df['Precipitation_mm'], bins=15, color='lightblue', edgecolor='black', alpha=0.7)
    axes[2, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[2, 0].plot(df['Precipitation_mm'], np.zeros_like(df['Precipitation_mm']), '|', color='red', markersize=10)
    axes[2, 0].set_title('Precipitation Distribution with Rug Plot', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Precipitation (mm)')
    axes[2, 0].set_ylabel('Frequency')
    
    # 8. Scatter Plot - Temperature vs Yield per Hectare
    axes[2, 1].scatter(df['Temperature_Celsius'], df['Yield_per_Hectare'], alpha=0.6, color='red')
    axes[2, 1].set_title('Temperature vs Yield per Hectare (Scatter Plot)', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Temperature (°C)')
    axes[2, 1].set_ylabel('Yield per Hectare (Tonnes)')
    
    # 9. Line Plot - Market Price Trends by Year
    price_by_year = df.groupby('Year')['Market_Price_per_Tonne'].mean()
    axes[2, 2].plot(price_by_year.index, price_by_year.values, marker='s', linewidth=2, markersize=8)
    axes[2, 2].set_title('Market Price Trends by Year (Line Plot)', fontsize=12, fontweight='bold')
    axes[2, 2].set_xlabel('Year')
    axes[2, 2].set_ylabel('Average Market Price per Tonne ($)')
    
    plt.tight_layout()
    plt.savefig('continuous_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate summary statistics
def generate_summary_statistics():
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nCATEGORICAL VARIABLES SUMMARY:")
    print("-" * 40)
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Unique values: {df[col].nunique()}")
    
    print("\n\nCONTINUOUS VARIABLES SUMMARY:")
    print("-" * 40)
    print(df[continuous_cols].describe())

# Execute the analysis
if __name__ == "__main__":
    print("Starting Univariate Analysis...")
    print("="*60)
    
    # Generate summary statistics
    generate_summary_statistics()
    
    # Create categorical visualizations
    print("\nCreating categorical data visualizations...")
    create_categorical_visualizations()
    
    # Create continuous visualizations
    print("\nCreating continuous data visualizations...")
    create_continuous_visualizations()
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

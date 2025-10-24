# Real Estate Property Listings Analysis

A comprehensive Python-based analysis system for real estate property data covering dataset exploration, visualization, and interactive analytics.

## Project Overview

This project implements five course outcomes (CO1-CO5) for real estate data analysis:

### CO1: Dataset Attributes Analysis
- Identifies and analyzes key dataset attributes (price, area, location, type)
- Provides statistical summaries and data distributions
- Visualizes numerical and categorical attributes

### CO2: Price vs Area Analysis
- Scatter plots showing price-area relationships
- Violin plots for distribution analysis
- Correlation analysis and regression trends
- Box plots by location and property type

### CO3: Property Type Hierarchy
- TreeMap visualizations showing property hierarchies
- Multi-level treemaps (Type → Location → Price Range)
- Total value and count-based representations

### CO4: Spatial Visualization
- Interactive maps with property markers
- Heat maps showing price density
- Circle markers sized by property value
- Static geographic scatter plots

### CO5: Interactive Price Analyzer
- Power BI-ready dataset export
- Interactive Plotly dashboards
- Summary statistics and KPI visualizations
- Comprehensive Power BI setup instructions

## Installation

1. Install Python 3.8 or higher
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the complete analysis:

```bash
python real_estate_analysis.py
```

This will:
1. Generate a synthetic real estate dataset (500 properties)
2. Create all visualizations for CO1-CO5
3. Save outputs to the `output` directory

## Output Files

The script generates the following outputs in the `output/` directory:

### Datasets
- `real_estate_dataset.csv` - Main dataset
- `real_estate_data_for_powerbi.csv` - Power BI ready dataset

### CO1 Outputs
- `CO1_attributes_distribution.png` - Numerical attribute distributions
- `CO1_categorical_attributes.png` - Property type and location distributions

### CO2 Outputs
- `CO2_scatter_plot.png` - Price vs Area scatter plot
- `CO2_violin_plots.png` - Distribution violin plots
- `CO2_comprehensive_analysis.png` - Combined analysis with correlations

### CO3 Outputs
- `CO3_treemap_count.png` - Property count hierarchy
- `CO3_treemap_value.png` - Property value hierarchy
- `CO3_treemap_squarify.png` - Alternative treemap visualization
- `CO3_treemap_multilevel.png` - Multi-level hierarchy

### CO4 Outputs
- `CO4_map_markers.html` - Interactive marker map
- `CO4_map_heatmap.html` - Price density heatmap
- `CO4_map_circles.html` - Circle markers by price
- `CO4_static_maps.png` - Static geographic visualizations
- `CO4_interactive_map.html` - Plotly interactive map

### CO5 Outputs
- `CO5_interactive_dashboard.html` - Multi-panel dashboard
- `CO5_price_analyzer_interactive.html` - Price analyzer with filters
- `CO5_summary_statistics.png` - Market summary statistics
- `CO5_PowerBI_Instructions.txt` - Power BI setup guide

## Dataset Attributes

The synthetic dataset includes the following attributes:

- **Property_ID**: Unique identifier
- **Type**: Property type (Apartment, Villa, House, Condo, Townhouse)
- **Price**: Property price in USD
- **Area_SqFt**: Area in square feet
- **Location**: City location
- **Latitude/Longitude**: Geographic coordinates
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms
- **Year_Built**: Construction year
- **Parking_Spaces**: Number of parking spaces
- **Price_Per_SqFt**: Calculated price per square foot

## Technologies Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations
- **folium**: Interactive maps
- **squarify**: Treemap visualizations

## Power BI Integration

For CO5, import the `real_estate_data_for_powerbi.csv` file into Power BI Desktop and follow the instructions in `CO5_PowerBI_Instructions.txt` to create an interactive dashboard.

## Project Structure

```
task-10-dv/
├── real_estate_analysis.py      # Main analysis script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── output/                       # Generated outputs
    ├── *.csv                     # Dataset files
    ├── *.png                     # Static visualizations
    ├── *.html                    # Interactive visualizations
    └── *.txt                     # Documentation
```

## License

This project is for educational purposes.


# Task 8: Spatial and Geospatial Data Analysis - Tamil Nadu Cities and Towns

## Overview
This project analyzes and visualizes spatial and geospatial data for cities and towns in Tamil Nadu, India, using population statistics from census years 1991, 2001, and 2011.

## Dataset Information
- **Source**: Cities and Towns in TN - Population Statistics (Kaggle)
- **Columns**: 
  - Name of city/town
  - Status (City/Town)
  - District
  - Population statistics (1991-03-01, 2001-03-01, 2011-03-01)
  - Geographic coordinates (Latitude, Longitude)

## Files
- `spatial_analysis.py` - Main Python script for analysis and visualization
- `tamil_nadu_spatial_analysis.png` - Output visualization image

## Key Features

### 1. **Geographic Distribution Map (2011 Population)**
   - Bubble size represents population
   - Color intensity shows population density
   - Major cities are labeled
   - Map projection: Latitude/Longitude coordinate system

### 2. **Population Growth Rate Map (1991-2011)**
   - Shows percentage change over 20 years
   - Color coding: Red (low) to Green (high growth)
   - Top growth cities are annotated

### 3. **District-wise Analysis**
   - Top 10 districts by total population
   - Horizontal bar chart with population values

### 4. **Temporal Population Trends**
   - Line graphs showing population change over time
   - Top 10 cities tracked across three census years

### 5. **Spatial Distribution by Status**
   - Cities (circles) vs Towns (triangles)
   - Geographic clustering patterns

### 6. **Growth Rate Distribution**
   - Spatial visualization with growth indicators
   - Arrow height represents growth magnitude
   - Color coding: Green (high), Orange (moderate), Red (low/negative)

## Key Insights

1. **Total analyzed**: 25 cities and towns
2. **Most populous city (2011)**: Chennai (4,681,087)
3. **Highest growth rate**: Vellore (138.91%)
4. **Average growth (1991-2011)**: 40.13%
5. **Total urban population (2011)**: 12,627,048
6. **Geographic spread**: 
   - Latitude: 8.0° to 13.5°
   - Longitude: 76.5° to 80.5°

## Requirements
```
pandas
numpy
matplotlib
seaborn
```

## How to Run
```bash
python spatial_analysis.py
```

## Output
The script generates:
1. Console output with statistical analysis
2. High-resolution PNG image (300 DPI) with 6 different visualizations

## Map Projections Used
- **Geographic Coordinate System**: WGS84 (Latitude/Longitude)
- **Projection Type**: Equirectangular (for display purposes)
- **Coverage**: Tamil Nadu state boundaries (approximate)

## Spatial Analysis Techniques
1. Point pattern analysis (geographic distribution)
2. Choropleth-style visualization (growth rates)
3. Temporal-spatial correlation
4. Cluster analysis by administrative units
5. Multi-variate spatial representation

## Author
Created for CO4, S3 - Spatial and Geospatial Data Analysis



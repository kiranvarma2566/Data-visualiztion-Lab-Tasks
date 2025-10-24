"""
Real Estate Property Listings Analysis - OPTIMIZED VERSION
All CO1-CO5 functionality in compact, efficient code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import squarify
import os

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
os.makedirs('output', exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_fig(name, dpi=300):
    """Save figure with consistent settings"""
    plt.tight_layout()
    plt.savefig(f'output/{name}.png', dpi=dpi, bbox_inches='tight')
    print(f"[OK] Saved: output/{name}.png")
    plt.close()

def format_currency(x, p):
    """Format currency for plots"""
    return f'${x/1e6:.1f}M'

# ============================================================================
# DATA GENERATION
# ============================================================================

def create_dataset():
    """Generate real estate dataset"""
    np.random.seed(42)
    n = 500
    
    types = ['Apartment', 'Villa', 'House', 'Condo', 'Townhouse']
    locations = {
        'New York': (40.7128, -74.0060), 'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298), 'Houston': (29.7604, -95.3698),
        'Phoenix': (33.4484, -112.0740), 'Philadelphia': (39.9526, -75.1652),
        'San Antonio': (29.4241, -98.4936), 'San Diego': (32.7157, -117.1611),
        'Dallas': (32.7767, -96.7970), 'San Jose': (37.3382, -121.8863)
    }
    
    area_ranges = {'Apartment': (900, 300), 'Villa': (3500, 800), 'House': (2000, 500),
                   'Condo': (1200, 400), 'Townhouse': (1800, 450)}
    
    price_per_sqft = {'New York': 800, 'Los Angeles': 650, 'Chicago': 350, 'Houston': 250,
                      'Phoenix': 280, 'Philadelphia': 320, 'San Antonio': 200,
                      'San Diego': 600, 'Dallas': 300, 'San Jose': 850}
    
    type_mult = {'Apartment': 0.9, 'Villa': 1.3, 'House': 1.0, 'Condo': 0.95, 'Townhouse': 1.05}
    
    data = []
    for i in range(n):
        ptype = str(np.random.choice(types, p=[0.30, 0.15, 0.25, 0.20, 0.10]))
        loc = str(np.random.choice(list(locations.keys())))
        
        area = max(500, np.random.normal(*area_ranges[ptype]))
        price = area * price_per_sqft[loc] * type_mult[ptype] * np.random.uniform(0.85, 1.15)
        
        data.append({
            'Property_ID': f'PROP_{i+1:04d}', 'Type': ptype, 'Price': round(price, 2),
            'Area_SqFt': round(area, 2), 'Location': loc,
            'Latitude': round(locations[loc][0] + np.random.uniform(-0.5, 0.5), 4),
            'Longitude': round(locations[loc][1] + np.random.uniform(-0.5, 0.5), 4),
            'Bedrooms': max(1, int(area / 500)), 'Bathrooms': max(1, int(area / 667)),
            'Year_Built': np.random.randint(1950, 2024),
            'Parking_Spaces': np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.35, 0.15]),
            'Price_Per_SqFt': round(price/area, 2)
        })
    
    return pd.DataFrame(data)

# ============================================================================
# CO1: DATASET ATTRIBUTES
# ============================================================================

def co1_analysis(df):
    """CO1: Identify and analyze dataset attributes"""
    print("\n" + "="*80)
    print("CO1: DATASET ATTRIBUTES ANALYSIS")
    print("="*80)
    
    # Print statistics
    for title, content in [
        ("Dataset Shape", f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"),
        ("Data Types", df.dtypes), ("First 5 Records", df.head()),
        ("Statistical Summary", df.describe()), ("Missing Values", df.isnull().sum()),
        ("Property Type", df['Type'].value_counts()), ("Location", df['Location'].value_counts())
    ]:
        print(f"\n{title}:\n{content}\n")
    
    # Numerical attributes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    df[['Price', 'Area_SqFt', 'Bedrooms', 'Bathrooms']].hist(
        bins=30, ax=axes, color='steelblue', edgecolor='black')
    plt.suptitle('CO1: Distribution of Key Numerical Attributes', fontsize=16, fontweight='bold')
    save_fig('CO1_attributes_distribution')
    
    # Categorical attributes
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    df['Type'].value_counts().plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Property Type Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Property Type', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    df['Location'].value_counts().plot(kind='barh', ax=axes[1], color='lightblue', edgecolor='black')
    axes[1].set_title('Location Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Count', fontsize=12)
    save_fig('CO1_categorical_attributes')

# ============================================================================
# CO2: PRICE VS AREA ANALYSIS
# ============================================================================

def co2_analysis(df):
    """CO2: Analyze price vs area using scatter and violin plots"""
    print("\n" + "="*80)
    print("CO2: PRICE VS AREA ANALYSIS")
    print("="*80)
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(14, 8))
    types = df['Type'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
    
    for ptype, color in zip(types, colors):
        mask = df['Type'] == ptype
        ax.scatter(df[mask]['Area_SqFt'], df[mask]['Price'], label=ptype, 
                  alpha=0.6, s=100, color=color, edgecolors='black')
    
    z = np.polyfit(df['Area_SqFt'], df['Price'], 1)
    p = np.poly1d(z)
    r2 = np.corrcoef(df['Area_SqFt'], df['Price'])[0,1]**2
    ax.plot(df['Area_SqFt'].sort_values(), p(df['Area_SqFt'].sort_values()), 
           "r--", linewidth=2, label=f'Trend Line (R²={r2:.3f})')
    
    ax.set_xlabel('Area (Square Feet)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax.set_title('CO2: Price vs Area - Scatter Plot by Property Type', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    save_fig('CO2_scatter_plot')
    
    # Violin plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.violinplot(data=df, x='Type', y='Price', ax=axes[0], hue='Type', palette='Set3', legend=False)
    axes[0].set_title('Price Distribution by Property Type', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    
    sns.violinplot(data=df, x='Type', y='Area_SqFt', ax=axes[1], hue='Type', palette='Set2', legend=False)
    axes[1].set_title('Area Distribution by Property Type', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    plt.suptitle('CO2: Violin Plots - Price and Area Analysis', fontsize=16, fontweight='bold')
    save_fig('CO2_violin_plots')
    
    # Comprehensive analysis
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    sns.boxplot(data=df, y='Location', x='Price', ax=axes[0,0], hue='Location', palette='coolwarm', legend=False)
    axes[0,0].set_title('Price by Location', fontsize=12, fontweight='bold')
    axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    
    sns.boxplot(data=df, y='Location', x='Area_SqFt', ax=axes[0,1], hue='Location', palette='viridis', legend=False)
    axes[0,1].set_title('Area by Location', fontsize=12, fontweight='bold')
    
    for ptype, color in zip(types, colors):
        mask = df['Type'] == ptype
        axes[1,0].scatter(df[mask]['Area_SqFt'], df[mask]['Price_Per_SqFt'], 
                         label=ptype, alpha=0.6, s=80, color=color, edgecolors='black')
    axes[1,0].set_title('Price per SqFt vs Area', fontsize=12, fontweight='bold')
    axes[1,0].legend(fontsize=9)
    axes[1,0].grid(True, alpha=0.3)
    
    corr = df[['Price', 'Area_SqFt', 'Bedrooms', 'Bathrooms', 'Year_Built', 'Price_Per_SqFt']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.suptitle('CO2: Comprehensive Price vs Area Analysis', fontsize=16, fontweight='bold')
    save_fig('CO2_comprehensive_analysis')

# ============================================================================
# CO3: TREEMAP VISUALIZATIONS
# ============================================================================

def co3_analysis(df):
    """CO3: Property type hierarchy using TreeMap"""
    print("\n" + "="*80)
    print("CO3: PROPERTY TYPE HIERARCHY - TREEMAP")
    print("="*80)
    
    # TreeMap 1: Count
    data = df.groupby(['Type', 'Location']).size().reset_index(name='Count')
    fig = px.treemap(data, path=['Type', 'Location'], values='Count',
                     title='CO3: Property Type Hierarchy - Count by Type and Location',
                     color='Count', color_continuous_scale='Viridis', height=700)
    fig.update_layout(font=dict(size=14, family="Arial Black"))
    fig.write_image('output/CO3_treemap_count.png', width=1600, height=900)
    print("[OK] Saved: output/CO3_treemap_count.png")
    
    # TreeMap 2: Value
    data = df.groupby(['Type', 'Location'])['Price'].sum().reset_index(name='Total_Value')
    fig = px.treemap(data, path=['Type', 'Location'], values='Total_Value',
                     title='CO3: Property Type Hierarchy - Total Value by Type and Location',
                     color='Total_Value', color_continuous_scale='RdYlGn', height=700)
    fig.update_traces(textinfo="label+value+percent parent")
    fig.write_image('output/CO3_treemap_value.png', width=1600, height=900)
    print("[OK] Saved: output/CO3_treemap_value.png")
    
    # TreeMap 3: Squarify
    type_sum = df.groupby('Type').agg({'Price': 'sum', 'Property_ID': 'count'}).reset_index()
    type_sum.columns = ['Type', 'Total_Value', 'Count']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    labels = [f"{r['Type']}\n{r['Count']} properties\n${r['Total_Value']/1e6:.1f}M" 
              for _, r in type_sum.iterrows()]
    squarify.plot(sizes=type_sum['Total_Value'], label=labels, alpha=0.8,
                  color=plt.cm.Set3(np.linspace(0, 1, len(type_sum))),
                  text_kwargs={'fontsize': 14, 'weight': 'bold'}, ax=ax)
    ax.set_title('CO3: Property Type Hierarchy - TreeMap (Total Market Value)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    save_fig('CO3_treemap_squarify')
    
    # TreeMap 4: Multi-level
    df['Price_Range'] = pd.cut(df['Price'], bins=[0, 500000, 1000000, 2000000, float('inf')],
                                labels=['<$500K', '$500K-$1M', '$1M-$2M', '>$2M'])
    data = df.groupby(['Type', 'Price_Range', 'Location']).size().reset_index(name='Count')
    fig = px.treemap(data, path=['Type', 'Price_Range', 'Location'], values='Count',
                     title='CO3: Multi-Level Property Hierarchy - Type → Price Range → Location',
                     color='Count', color_continuous_scale='Plasma', height=700)
    fig.write_image('output/CO3_treemap_multilevel.png', width=1600, height=900)
    print("[OK] Saved: output/CO3_treemap_multilevel.png")

# ============================================================================
# CO4: SPATIAL VISUALIZATION
# ============================================================================

def co4_analysis(df):
    """CO4: Visualize property distribution on map"""
    print("\n" + "="*80)
    print("CO4: SPATIAL VISUALIZATION - PROPERTY DISTRIBUTION MAP")
    print("="*80)
    
    center = [df['Latitude'].mean(), df['Longitude'].mean()]
    colors_map = {'Apartment': 'blue', 'Villa': 'red', 'House': 'green', 
                  'Condo': 'orange', 'Townhouse': 'purple'}
    
    # Map 1: Markers
    m = folium.Map(location=center, zoom_start=4, tiles='OpenStreetMap')
    cluster = MarkerCluster().add_to(m)
    
    for _, row in df.iterrows():
        popup = f"<b>{row['Property_ID']}</b><br>Type: {row['Type']}<br>Price: ${row['Price']:,.0f}<br>Area: {row['Area_SqFt']:.0f} sq ft"
        folium.Marker([row['Latitude'], row['Longitude']], popup=popup,
                     icon=folium.Icon(color=colors_map.get(row['Type'], 'gray'), icon='home', prefix='fa'),
                     tooltip=f"{row['Type']} - ${row['Price']:,.0f}").add_to(cluster)
    m.save('output/CO4_map_markers.html')
    print("[OK] Saved: output/CO4_map_markers.html")
    
    # Map 2: Heatmap
    m = folium.Map(location=center, zoom_start=4, tiles='CartoDB positron')
    heat_data = [[r['Latitude'], r['Longitude'], r['Price']/1e6] for _, r in df.iterrows()]
    HeatMap(heat_data, radius=15, blur=25).add_to(m)
    m.save('output/CO4_map_heatmap.html')
    print("[OK] Saved: output/CO4_map_heatmap.html")
    
    # Map 3: Circles
    m = folium.Map(location=center, zoom_start=4)
    for _, row in df.iterrows():
        folium.CircleMarker([row['Latitude'], row['Longitude']], radius=row['Price']/200000,
                           popup=f"{row['Type']}<br>${row['Price']:,.0f}",
                           color=colors_map.get(row['Type'], 'gray'), fill=True,
                           fillColor=colors_map.get(row['Type'], 'gray'), fillOpacity=0.6).add_to(m)
    m.save('output/CO4_map_circles.html')
    print("[OK] Saved: output/CO4_map_circles.html")
    
    # Static maps
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    types = df['Type'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
    
    for ptype, color in zip(types, colors):
        mask = df['Type'] == ptype
        axes[0].scatter(df[mask]['Longitude'], df[mask]['Latitude'], label=ptype,
                       alpha=0.6, s=df[mask]['Price']/5000, color=color, edgecolors='black', linewidth=0.5)
    axes[0].set_title('CO4: Property Distribution Map (Size = Price)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    hexbin = axes[1].hexbin(df['Longitude'], df['Latitude'], C=df['Price'], 
                           gridsize=20, cmap='YlOrRd', reduce_C_function=np.mean, mincnt=1)
    axes[1].set_title('CO4: Property Price Density (Hexbin)', fontsize=14, fontweight='bold')
    plt.colorbar(hexbin, ax=axes[1], label='Average Price ($)')
    save_fig('CO4_static_maps')
    
    # Interactive Plotly
    fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', color='Type', size='Price',
                           hover_name='Property_ID', zoom=3, height=800,
                           title='CO4: Interactive Property Distribution Map')
    fig.update_layout(mapbox_style='open-street-map', margin={"r":0,"t":40,"l":0,"b":0})
    fig.write_html('output/CO4_interactive_map.html')
    print("[OK] Saved: output/CO4_interactive_map.html")

# ============================================================================
# CO5: INTERACTIVE PRICE ANALYZER
# ============================================================================

def co5_analysis(df):
    """CO5: Build interactive real estate price analyzer"""
    print("\n" + "="*80)
    print("CO5: INTERACTIVE REAL ESTATE PRICE ANALYZER")
    print("="*80)
    
    # Save for Power BI
    df.to_csv('output/real_estate_data_for_powerbi.csv', index=False)
    print("[OK] Saved: output/real_estate_data_for_powerbi.csv")
    
    # Dashboard
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=('Price by Type', 'Properties by Location', 
                                     'Price vs Area', 'Avg Price per SqFt'),
                       specs=[[{'type': 'box'}, {'type': 'bar'}],
                             [{'type': 'scatter'}, {'type': 'bar'}]])
    
    for ptype in df['Type'].unique():
        fig.add_trace(go.Box(y=df[df['Type']==ptype]['Price'], name=ptype), row=1, col=1)
    
    loc_counts = df['Location'].value_counts()
    fig.add_trace(go.Bar(x=loc_counts.index, y=loc_counts.values, showlegend=False), row=1, col=2)
    
    for ptype in df['Type'].unique():
        mask = df['Type'] == ptype
        fig.add_trace(go.Scatter(x=df[mask]['Area_SqFt'], y=df[mask]['Price'], 
                                mode='markers', name=ptype), row=2, col=1)
    
    avg_price = df.groupby('Type')['Price_Per_SqFt'].mean().sort_values()
    fig.add_trace(go.Bar(x=avg_price.values, y=avg_price.index, 
                        orientation='h', showlegend=False), row=2, col=2)
    
    fig.update_layout(height=900, title_text="CO5: Interactive Real Estate Price Analyzer Dashboard")
    fig.write_html('output/CO5_interactive_dashboard.html')
    print("[OK] Saved: output/CO5_interactive_dashboard.html")
    
    # Price analyzer with filters
    fig = go.Figure()
    for ptype in df['Type'].unique():
        mask = df['Type'] == ptype
        fig.add_trace(go.Scatter(x=df[mask]['Area_SqFt'], y=df[mask]['Price'],
                                mode='markers', name=ptype, marker=dict(size=10, opacity=0.7)))
    fig.update_layout(title='CO5: Interactive Price Analyzer', height=700, hovermode='closest')
    fig.write_html('output/CO5_price_analyzer_interactive.html')
    print("[OK] Saved: output/CO5_price_analyzer_interactive.html")
    
    # Power BI instructions
    instructions = """# Power BI Setup Instructions

1. Open Power BI Desktop
2. Get Data → CSV → Select 'real_estate_data_for_powerbi.csv'
3. Create these visualizations:
   - KPI Cards: Total Properties, Avg Price, Total Market Value
   - Slicers: Property Type, Location, Price Range
   - Charts: Clustered Column, Scatter, Map, TreeMap, Pie
   
4. DAX Measures:
   Total Properties = COUNTROWS('real_estate_data')
   Average Price = AVERAGE('real_estate_data'[Price])
   Total Market Value = SUM('real_estate_data'[Price])
   Premium Properties = CALCULATE(COUNTROWS('real_estate_data'), 'real_estate_data'[Price] > 1000000)
"""
    with open('output/CO5_PowerBI_Instructions.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)
    print("[OK] Saved: output/CO5_PowerBI_Instructions.txt")
    
    # Summary statistics
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    df.groupby('Type')['Price'].mean().sort_values().plot(kind='barh', ax=axes[0,0], color='skyblue', edgecolor='black')
    axes[0,0].set_title('Average Price by Property Type', fontsize=13, fontweight='bold')
    axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    
    df['Type'].value_counts().plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%', 
                                   colors=plt.cm.Set3(np.linspace(0, 1, len(df['Type'].unique()))))
    axes[0,1].set_title('Property Type Distribution', fontsize=13, fontweight='bold')
    
    df.groupby('Location')['Price'].mean().sort_values().plot(kind='barh', ax=axes[1,0], color='lightcoral', edgecolor='black')
    axes[1,0].set_title('Average Price by Location', fontsize=13, fontweight='bold')
    axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    
    pd.cut(df['Price'], bins=5).value_counts().sort_index().plot(kind='bar', ax=axes[1,1], color='lightgreen', edgecolor='black')
    axes[1,1].set_title('Price Range Distribution', fontsize=13, fontweight='bold')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('CO5: Real Estate Market Summary Statistics', fontsize=16, fontweight='bold')
    save_fig('CO5_summary_statistics')

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution - all CO1-CO5 analyses"""
    print("\n" + "="*80)
    print("         REAL ESTATE PROPERTY LISTINGS ANALYSIS")
    print("="*80)
    print("  CO1: Dataset Attributes Analysis")
    print("  CO2: Price vs Area Analysis (Scatter & Violin Plots)")
    print("  CO3: Property Type Hierarchy (TreeMap)")
    print("  CO4: Spatial Visualization (Maps)")
    print("  CO5: Interactive Price Analyzer (Power BI)")
    print("="*80 + "\n")
    
    # Create and save dataset
    print("Creating real estate dataset...")
    df = create_dataset()
    df.to_csv('output/real_estate_dataset.csv', index=False)
    print(f"[OK] Dataset created: {len(df)} properties")
    print("[OK] Saved: output/real_estate_dataset.csv\n")
    
    # Execute all analyses
    co1_analysis(df)
    co2_analysis(df)
    co3_analysis(df)
    co4_analysis(df)
    co5_analysis(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nAll outputs saved in 'output/' directory")
    print("  - 2 CSV files (datasets)")
    print("  - 11 PNG files (static visualizations)")
    print("  - 6 HTML files (interactive maps/dashboards)")
    print("  - 1 TXT file (Power BI instructions)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()


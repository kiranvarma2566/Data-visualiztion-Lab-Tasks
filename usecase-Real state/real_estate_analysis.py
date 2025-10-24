"""
Real Estate Property Listings Analysis
CO1-CO5: Comprehensive analysis of property data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
import squarify
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
os.makedirs('output', exist_ok=True)

# ============================================================================
# CO1: CREATE AND IDENTIFY DATASET ATTRIBUTES
# ============================================================================

def create_real_estate_dataset():
    """
    Create a synthetic real estate dataset with attributes:
    - price, area, location, type, bedrooms, bathrooms, etc.
    """
    np.random.seed(42)
    n_samples = 500
    
    # Property types
    property_types = ['Apartment', 'Villa', 'House', 'Condo', 'Townhouse']
    
    # Locations (cities with coordinates)
    locations = {
        'New York': (40.7128, -74.0060),
        'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298),
        'Houston': (29.7604, -95.3698),
        'Phoenix': (33.4484, -112.0740),
        'Philadelphia': (39.9526, -75.1652),
        'San Antonio': (29.4241, -98.4936),
        'San Diego': (32.7157, -117.1611),
        'Dallas': (32.7767, -96.7970),
        'San Jose': (37.3382, -121.8863)
    }
    
    data = []
    
    for i in range(n_samples):
        location = str(np.random.choice(list(locations.keys())))
        prop_type = str(np.random.choice(property_types, p=[0.30, 0.15, 0.25, 0.20, 0.10]))
        
        # Area varies by property type
        if prop_type == 'Apartment':
            area = np.random.normal(900, 300)
        elif prop_type == 'Villa':
            area = np.random.normal(3500, 800)
        elif prop_type == 'House':
            area = np.random.normal(2000, 500)
        elif prop_type == 'Condo':
            area = np.random.normal(1200, 400)
        else:  # Townhouse
            area = np.random.normal(1800, 450)
        
        area = max(500, area)  # Minimum area
        
        # Price calculation based on area, location, and type
        base_price_per_sqft = {
            'New York': 800,
            'Los Angeles': 650,
            'Chicago': 350,
            'Houston': 250,
            'Phoenix': 280,
            'Philadelphia': 320,
            'San Antonio': 200,
            'San Diego': 600,
            'Dallas': 300,
            'San Jose': 850
        }
        
        type_multiplier = {
            'Apartment': 0.9,
            'Villa': 1.3,
            'House': 1.0,
            'Condo': 0.95,
            'Townhouse': 1.05
        }
        
        price_per_sqft = base_price_per_sqft[location] * type_multiplier[prop_type]
        price = area * price_per_sqft * np.random.uniform(0.85, 1.15)
        
        # Bedrooms and bathrooms
        bedrooms = max(1, int(area / 500))
        bathrooms = max(1, int(bedrooms * 0.75))
        
        # Add some randomness to coordinates
        lat = locations[location][0] + np.random.uniform(-0.5, 0.5)
        lon = locations[location][1] + np.random.uniform(-0.5, 0.5)
        
        # Year built
        year_built = np.random.randint(1950, 2024)
        
        # Parking spaces
        parking = np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.35, 0.15])
        
        data.append({
            'Property_ID': f'PROP_{i+1:04d}',
            'Type': prop_type,
            'Price': round(price, 2),
            'Area_SqFt': round(area, 2),
            'Location': location,
            'Latitude': round(lat, 4),
            'Longitude': round(lon, 4),
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Year_Built': year_built,
            'Parking_Spaces': parking,
            'Price_Per_SqFt': round(price/area, 2)
        })
    
    df = pd.DataFrame(data)
    return df

def analyze_dataset_attributes(df):
    """CO1: Identify and analyze dataset attributes"""
    print("=" * 80)
    print("CO1: DATASET ATTRIBUTES ANALYSIS")
    print("=" * 80)
    
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n2. Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n3. First 5 Records:")
    print(df.head())
    
    print("\n4. Statistical Summary:")
    print(df.describe())
    
    print("\n5. Missing Values:")
    print(df.isnull().sum())
    
    print("\n6. Property Type Distribution:")
    print(df['Type'].value_counts())
    
    print("\n7. Location Distribution:")
    print(df['Location'].value_counts())
    
    # Create visualization of attributes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Numerical attributes distribution
    df[['Price', 'Area_SqFt', 'Bedrooms', 'Bathrooms']].hist(
        bins=30, ax=axes, color='steelblue', edgecolor='black'
    )
    
    plt.suptitle('CO1: Distribution of Key Numerical Attributes', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('output/CO1_attributes_distribution.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: output/CO1_attributes_distribution.png")
    plt.close()
    
    # Categorical attributes
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Property Type
    type_counts = df['Type'].value_counts()
    axes[0].bar(type_counts.index, type_counts.values, color='coral', edgecolor='black')
    axes[0].set_title('Property Type Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Property Type', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Location
    location_counts = df['Location'].value_counts()
    axes[1].barh(location_counts.index, location_counts.values, color='lightblue', edgecolor='black')
    axes[1].set_title('Location Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Count', fontsize=12)
    axes[1].set_ylabel('Location', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/CO1_categorical_attributes.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: output/CO1_categorical_attributes.png")
    plt.close()

# ============================================================================
# CO2: ANALYZE PRICE VS AREA (SCATTER AND VIOLIN PLOTS)
# ============================================================================

def analyze_price_vs_area(df):
    """CO2: Analyze price vs area using scatter and violin plots"""
    print("\n" + "=" * 80)
    print("CO2: PRICE VS AREA ANALYSIS")
    print("=" * 80)
    
    # Scatter Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create scatter plot with property type colors
    property_types = df['Type'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(property_types)))
    
    for prop_type, color in zip(property_types, colors):
        mask = df['Type'] == prop_type
        ax.scatter(df[mask]['Area_SqFt'], df[mask]['Price'], 
                  label=prop_type, alpha=0.6, s=100, color=color, edgecolors='black')
    
    # Add regression line
    z = np.polyfit(df['Area_SqFt'], df['Price'], 1)
    p = np.poly1d(z)
    ax.plot(df['Area_SqFt'].sort_values(), 
            p(df['Area_SqFt'].sort_values()), 
            "r--", linewidth=2, label=f'Trend Line (R²={np.corrcoef(df["Area_SqFt"], df["Price"])[0,1]**2:.3f})')
    
    ax.set_xlabel('Area (Square Feet)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
    ax.set_title('CO2: Price vs Area - Scatter Plot by Property Type', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    plt.tight_layout()
    plt.savefig('output/CO2_scatter_plot.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: output/CO2_scatter_plot.png")
    plt.close()
    
    # Violin Plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Violin plot: Price by Property Type
    sns.violinplot(data=df, x='Type', y='Price', ax=axes[0], palette='Set3')
    axes[0].set_title('Price Distribution by Property Type', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Property Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Violin plot: Area by Property Type
    sns.violinplot(data=df, x='Type', y='Area_SqFt', ax=axes[1], palette='Set2')
    axes[1].set_title('Area Distribution by Property Type', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Property Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Area (Square Feet)', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('CO2: Violin Plots - Price and Area Analysis', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('output/CO2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: output/CO2_violin_plots.png")
    plt.close()
    
    # Combined analysis with box plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Box plot: Price by Location
    sns.boxplot(data=df, y='Location', x='Price', ax=axes[0, 0], palette='coolwarm')
    axes[0, 0].set_title('Price Distribution by Location', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Price ($)', fontsize=11)
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Box plot: Area by Location
    sns.boxplot(data=df, y='Location', x='Area_SqFt', ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Area Distribution by Location', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Area (Square Feet)', fontsize=11)
    
    # Scatter: Price per SqFt vs Area
    for prop_type, color in zip(property_types, colors):
        mask = df['Type'] == prop_type
        axes[1, 0].scatter(df[mask]['Area_SqFt'], df[mask]['Price_Per_SqFt'], 
                          label=prop_type, alpha=0.6, s=80, color=color, edgecolors='black')
    axes[1, 0].set_xlabel('Area (Square Feet)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Price per SqFt ($)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Price per SqFt vs Area', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_data = df[['Price', 'Area_SqFt', 'Bedrooms', 'Bathrooms', 'Year_Built', 'Price_Per_SqFt']]
    corr_matrix = corr_data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
    axes[1, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.suptitle('CO2: Comprehensive Price vs Area Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('output/CO2_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: output/CO2_comprehensive_analysis.png")
    plt.close()

# ============================================================================
# CO3: PROPERTY TYPE HIERARCHY USING TREEMAP
# ============================================================================

def create_property_treemap(df):
    """CO3: Represent property type hierarchy using TreeMap"""
    print("\n" + "=" * 80)
    print("CO3: PROPERTY TYPE HIERARCHY - TREEMAP")
    print("=" * 80)
    
    # TreeMap 1: Count by Type and Location
    treemap_data = df.groupby(['Type', 'Location']).size().reset_index(name='Count')
    
    fig = px.treemap(treemap_data, 
                     path=['Type', 'Location'], 
                     values='Count',
                     title='CO3: Property Type Hierarchy - Count by Type and Location',
                     color='Count',
                     color_continuous_scale='Viridis',
                     height=700)
    
    fig.update_layout(font=dict(size=14, family="Arial Black"))
    fig.write_image('output/CO3_treemap_count.png', width=1600, height=900)
    print("\n[OK] Saved: output/CO3_treemap_count.png")
    
    # TreeMap 2: Total Value by Type and Location
    treemap_value = df.groupby(['Type', 'Location'])['Price'].sum().reset_index(name='Total_Value')
    
    fig = px.treemap(treemap_value, 
                     path=['Type', 'Location'], 
                     values='Total_Value',
                     title='CO3: Property Type Hierarchy - Total Value by Type and Location',
                     color='Total_Value',
                     color_continuous_scale='RdYlGn',
                     height=700)
    
    fig.update_traces(textinfo="label+value+percent parent")
    fig.update_layout(font=dict(size=14, family="Arial Black"))
    fig.write_image('output/CO3_treemap_value.png', width=1600, height=900)
    print("[OK] Saved: output/CO3_treemap_value.png")
    
    # TreeMap 3: Using Squarify (Alternative visualization)
    type_summary = df.groupby('Type').agg({
        'Price': 'sum',
        'Property_ID': 'count'
    }).reset_index()
    type_summary.columns = ['Type', 'Total_Value', 'Count']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_summary)))
    
    # Create labels with information
    labels = [f"{row['Type']}\n{row['Count']} properties\n${row['Total_Value']/1e6:.1f}M" 
              for _, row in type_summary.iterrows()]
    
    squarify.plot(sizes=type_summary['Total_Value'], 
                  label=labels, 
                  alpha=0.8,
                  color=colors,
                  text_kwargs={'fontsize': 14, 'weight': 'bold'},
                  ax=ax)
    
    ax.set_title('CO3: Property Type Hierarchy - TreeMap (Total Market Value)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/CO3_treemap_squarify.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: output/CO3_treemap_squarify.png")
    plt.close()
    
    # TreeMap 4: Multi-level hierarchy (Type -> Location -> Price Range)
    df['Price_Range'] = pd.cut(df['Price'], 
                                bins=[0, 500000, 1000000, 2000000, float('inf')],
                                labels=['<$500K', '$500K-$1M', '$1M-$2M', '>$2M'])
    
    treemap_multi = df.groupby(['Type', 'Location', 'Price_Range']).size().reset_index(name='Count')
    
    fig = px.treemap(treemap_multi, 
                     path=['Type', 'Price_Range', 'Location'], 
                     values='Count',
                     title='CO3: Multi-Level Property Hierarchy - Type → Price Range → Location',
                     color='Count',
                     color_continuous_scale='Plasma',
                     height=700)
    
    fig.update_layout(font=dict(size=13, family="Arial"))
    fig.write_image('output/CO3_treemap_multilevel.png', width=1600, height=900)
    print("[OK] Saved: output/CO3_treemap_multilevel.png")

# ============================================================================
# CO4: SPATIAL VISUALIZATION - PROPERTY DISTRIBUTION ON MAP
# ============================================================================

def create_spatial_visualization(df):
    """CO4: Visualize property distribution on map"""
    print("\n" + "=" * 80)
    print("CO4: SPATIAL VISUALIZATION - PROPERTY DISTRIBUTION MAP")
    print("=" * 80)
    
    # Map 1: Interactive marker map with clusters
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    map_markers = folium.Map(location=[center_lat, center_lon], 
                             zoom_start=4,
                             tiles='OpenStreetMap')
    
    marker_cluster = MarkerCluster().add_to(map_markers)
    
    # Add markers for each property
    type_colors = {
        'Apartment': 'blue',
        'Villa': 'red',
        'House': 'green',
        'Condo': 'orange',
        'Townhouse': 'purple'
    }
    
    for _, row in df.iterrows():
        popup_text = f"""
        <b>Property ID:</b> {row['Property_ID']}<br>
        <b>Type:</b> {row['Type']}<br>
        <b>Price:</b> ${row['Price']:,.0f}<br>
        <b>Area:</b> {row['Area_SqFt']:.0f} sq ft<br>
        <b>Location:</b> {row['Location']}<br>
        <b>Bedrooms:</b> {row['Bedrooms']}<br>
        <b>Bathrooms:</b> {row['Bathrooms']}
        """
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=type_colors.get(row['Type'], 'gray'), 
                           icon='home', prefix='fa'),
            tooltip=f"{row['Type']} - ${row['Price']:,.0f}"
        ).add_to(marker_cluster)
    
    map_markers.save('output/CO4_map_markers.html')
    print("\n[OK] Saved: output/CO4_map_markers.html")
    
    # Map 2: Heatmap based on property prices
    map_heatmap = folium.Map(location=[center_lat, center_lon], 
                             zoom_start=4,
                             tiles='CartoDB positron')
    
    # Prepare heat data (latitude, longitude, weight=price)
    heat_data = [[row['Latitude'], row['Longitude'], row['Price']/1e6] 
                 for _, row in df.iterrows()]
    
    HeatMap(heat_data, 
            radius=15,
            blur=25,
            max_zoom=13,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1.0: 'red'}).add_to(map_heatmap)
    
    map_heatmap.save('output/CO4_map_heatmap.html')
    print("[OK] Saved: output/CO4_map_heatmap.html")
    
    # Map 3: Circle markers sized by price
    map_circles = folium.Map(location=[center_lat, center_lon], 
                            zoom_start=4,
                            tiles='OpenStreetMap')
    
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Price']/200000,  # Scale circle size by price
            popup=f"{row['Type']}<br>${row['Price']:,.0f}",
            color=type_colors.get(row['Type'], 'gray'),
            fill=True,
            fillColor=type_colors.get(row['Type'], 'gray'),
            fillOpacity=0.6
        ).add_to(map_circles)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin:0; font-weight:bold;">Property Types</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:blue"></i> Apartment</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:red"></i> Villa</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:green"></i> House</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:orange"></i> Condo</p>
    <p style="margin:5px 0;"><i class="fa fa-circle" style="color:purple"></i> Townhouse</p>
    <p style="margin:5px 0; font-size:12px;"><i>Circle size = Price</i></p>
    </div>
    '''
    map_circles.get_root().html.add_child(folium.Element(legend_html))
    
    map_circles.save('output/CO4_map_circles.html')
    print("[OK] Saved: output/CO4_map_circles.html")
    
    # Static visualization using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Scatter plot on map-like view
    for prop_type, color in zip(df['Type'].unique(), 
                                plt.cm.Set2(np.linspace(0, 1, len(df['Type'].unique())))):
        mask = df['Type'] == prop_type
        axes[0].scatter(df[mask]['Longitude'], df[mask]['Latitude'], 
                       label=prop_type, alpha=0.6, s=df[mask]['Price']/5000, 
                       color=color, edgecolors='black', linewidth=0.5)
    
    axes[0].set_xlabel('Longitude', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Latitude', fontsize=13, fontweight='bold')
    axes[0].set_title('CO4: Property Distribution Map (Size = Price)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Hexbin plot for density
    hexbin = axes[1].hexbin(df['Longitude'], df['Latitude'], 
                           C=df['Price'], gridsize=20, cmap='YlOrRd', 
                           reduce_C_function=np.mean, mincnt=1)
    axes[1].set_xlabel('Longitude', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Latitude', fontsize=13, fontweight='bold')
    axes[1].set_title('CO4: Property Price Density (Hexbin)', 
                     fontsize=14, fontweight='bold')
    
    cb = plt.colorbar(hexbin, ax=axes[1])
    cb.set_label('Average Price ($)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/CO4_static_maps.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: output/CO4_static_maps.png")
    plt.close()
    
    # Interactive Plotly map
    fig = px.scatter_mapbox(df, 
                           lat='Latitude', 
                           lon='Longitude',
                           color='Type',
                           size='Price',
                           hover_name='Property_ID',
                           hover_data={
                               'Price': ':$,.0f',
                               'Area_SqFt': ':,.0f',
                               'Location': True,
                               'Bedrooms': True,
                               'Latitude': False,
                               'Longitude': False
                           },
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           zoom=3,
                           height=800,
                           title='CO4: Interactive Property Distribution Map')
    
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig.write_html('output/CO4_interactive_map.html')
    print("[OK] Saved: output/CO4_interactive_map.html")

# ============================================================================
# CO5: INTERACTIVE REAL ESTATE PRICE ANALYZER
# ============================================================================

def create_powerbi_dataset(df):
    """CO5: Create dataset and interactive visualizations for Power BI"""
    print("\n" + "=" * 80)
    print("CO5: INTERACTIVE REAL ESTATE PRICE ANALYZER")
    print("=" * 80)
    
    # Save dataset for Power BI
    df.to_csv('output/real_estate_data_for_powerbi.csv', index=False)
    print("\n[OK] Saved: output/real_estate_data_for_powerbi.csv")
    print("  (Use this file to import into Power BI)")
    
    # Create interactive dashboard using Plotly (Power BI alternative)
    from plotly.subplots import make_subplots
    
    # Dashboard 1: Overview Dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price by Property Type', 
                       'Properties by Location',
                       'Price vs Area Relationship',
                       'Average Price per SqFt by Type'),
        specs=[[{'type': 'box'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # Box plot: Price by Type
    for prop_type in df['Type'].unique():
        fig.add_trace(
            go.Box(y=df[df['Type']==prop_type]['Price'], name=prop_type),
            row=1, col=1
        )
    
    # Bar chart: Count by Location
    location_counts = df['Location'].value_counts()
    fig.add_trace(
        go.Bar(x=location_counts.index, y=location_counts.values, 
               marker_color='lightblue', showlegend=False),
        row=1, col=2
    )
    
    # Scatter: Price vs Area
    for prop_type in df['Type'].unique():
        mask = df['Type'] == prop_type
        fig.add_trace(
            go.Scatter(x=df[mask]['Area_SqFt'], y=df[mask]['Price'],
                      mode='markers', name=prop_type,
                      marker=dict(size=8, opacity=0.6)),
            row=2, col=1
        )
    
    # Bar chart: Avg Price per SqFt by Type
    avg_price_sqft = df.groupby('Type')['Price_Per_SqFt'].mean().sort_values()
    fig.add_trace(
        go.Bar(x=avg_price_sqft.values, y=avg_price_sqft.index,
               orientation='h', marker_color='coral', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=900, showlegend=True,
                     title_text="CO5: Interactive Real Estate Price Analyzer Dashboard",
                     title_font_size=20)
    
    fig.write_html('output/CO5_interactive_dashboard.html')
    print("[OK] Saved: output/CO5_interactive_dashboard.html")
    
    # Dashboard 2: Price Analysis Dashboard
    fig2 = go.Figure()
    
    # Add traces for each property type
    for prop_type in df['Type'].unique():
        mask = df['Type'] == prop_type
        fig2.add_trace(go.Scatter(
            x=df[mask]['Area_SqFt'],
            y=df[mask]['Price'],
            mode='markers',
            name=prop_type,
            marker=dict(size=10, opacity=0.7),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Area: %{x:,.0f} sq ft<br>' +
                         'Price: $%{y:,.0f}<br>' +
                         '<extra></extra>'
        ))
    
    # Add dropdown for filtering by location
    buttons = [{'label': 'All Locations', 
                'method': 'update',
                'args': [{'visible': [True] * len(df['Type'].unique())}]}]
    
    for location in df['Location'].unique():
        visible = []
        for prop_type in df['Type'].unique():
            visible.append(True)
        buttons.append({
            'label': location,
            'method': 'update',
            'args': [{'x': [df[(df['Type']==pt) & (df['Location']==location)]['Area_SqFt'] 
                           for pt in df['Type'].unique()],
                     'y': [df[(df['Type']==pt) & (df['Location']==location)]['Price'] 
                           for pt in df['Type'].unique()]}]
        })
    
    fig2.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.17,
            'y': 1.15
        }],
        title='CO5: Interactive Price Analyzer with Location Filter',
        xaxis_title='Area (Square Feet)',
        yaxis_title='Price ($)',
        height=700,
        hovermode='closest'
    )
    
    fig2.write_html('output/CO5_price_analyzer_interactive.html')
    print("[OK] Saved: output/CO5_price_analyzer_interactive.html")
    
    # Create Power BI instruction document
    powerbi_instructions = """
# Power BI Dashboard Instructions

## Dataset: real_estate_data_for_powerbi.csv

### Recommended Visualizations for Power BI:

1. **KPI Cards:**
   - Total Properties
   - Average Price
   - Average Price per SqFt
   - Total Market Value

2. **Slicers (Filters):**
   - Property Type
   - Location
   - Price Range
   - Bedrooms
   - Year Built Range

3. **Visualizations:**

   a) **Clustered Column Chart:**
      - Axis: Property Type
      - Values: Average of Price
      - Legend: Location

   b) **Scatter Chart:**
      - X-Axis: Area_SqFt
      - Y-Axis: Price
      - Legend: Type
      - Size: Bedrooms

   c) **Map Visualization:**
      - Location: Latitude, Longitude
      - Size: Price
      - Legend: Type

   d) **Treemap:**
      - Group: Type, Location
      - Values: Price (Sum)

   e) **Line Chart:**
      - Axis: Year_Built
      - Values: Average Price
      - Legend: Type

   f) **Pie Chart:**
      - Legend: Type
      - Values: Count of Property_ID

   g) **Table:**
      - Columns: All fields
      - With conditional formatting on Price

4. **DAX Measures to Create:**

   ```DAX
   Total Properties = COUNTROWS('real_estate_data')
   
   Average Price = AVERAGE('real_estate_data'[Price])
   
   Total Market Value = SUM('real_estate_data'[Price])
   
   Avg Price per SqFt = AVERAGE('real_estate_data'[Price_Per_SqFt])
   
   Premium Properties = CALCULATE(COUNTROWS('real_estate_data'), 
                                   'real_estate_data'[Price] > 1000000)
   ```

### Dashboard Layout:
- Top Row: KPI Cards
- Left Sidebar: Slicers/Filters
- Main Area: Charts and visualizations
- Bottom: Detailed table

### Color Scheme Recommendations:
- Apartment: Blue (#4472C4)
- Villa: Red (#C55A11)
- House: Green (#70AD47)
- Condo: Orange (#FFC000)
- Townhouse: Purple (#7030A0)

### To Import into Power BI:
1. Open Power BI Desktop
2. Get Data → Text/CSV
3. Select 'real_estate_data_for_powerbi.csv'
4. Load the data
5. Create visualizations as recommended above
6. Add slicers for interactivity
7. Format and publish

"""
    
    with open('output/CO5_PowerBI_Instructions.txt', 'w', encoding='utf-8') as f:
        f.write(powerbi_instructions)
    
    print("[OK] Saved: output/CO5_PowerBI_Instructions.txt")
    
    # Create a summary statistics image
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Summary statistics by type
    summary_by_type = df.groupby('Type').agg({
        'Price': ['mean', 'median', 'count'],
        'Area_SqFt': 'mean',
        'Price_Per_SqFt': 'mean'
    }).round(2)
    
    # Bar chart: Average price by type
    avg_prices = df.groupby('Type')['Price'].mean().sort_values()
    axes[0, 0].barh(avg_prices.index, avg_prices.values, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Average Price ($)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Average Price by Property Type', fontsize=13, fontweight='bold')
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Pie chart: Property distribution
    type_counts = df['Type'].value_counts()
    axes[0, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(type_counts))))
    axes[0, 1].set_title('Property Type Distribution', fontsize=13, fontweight='bold')
    
    # Location analysis
    location_avg = df.groupby('Location')['Price'].mean().sort_values()
    axes[1, 0].barh(location_avg.index, location_avg.values, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Average Price ($)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Average Price by Location', fontsize=13, fontweight='bold')
    axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Price range distribution
    price_ranges = pd.cut(df['Price'], bins=5)
    price_range_counts = price_ranges.value_counts().sort_index()
    axes[1, 1].bar(range(len(price_range_counts)), price_range_counts.values, 
                   color='lightgreen', edgecolor='black')
    axes[1, 1].set_xlabel('Price Range', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Price Range Distribution', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(range(len(price_range_counts)))
    axes[1, 1].set_xticklabels([f'${int(interval.left/1000)}K-${int(interval.right/1000)}K' 
                                for interval in price_range_counts.index], 
                               rotation=45, ha='right', fontsize=9)
    
    plt.suptitle('CO5: Real Estate Market Summary Statistics', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('output/CO5_summary_statistics.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: output/CO5_summary_statistics.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n")
    print("=" * 80)
    print("         REAL ESTATE PROPERTY LISTINGS ANALYSIS")
    print("=" * 80)
    print("  CO1: Dataset Attributes Analysis")
    print("  CO2: Price vs Area Analysis (Scatter & Violin Plots)")
    print("  CO3: Property Type Hierarchy (TreeMap)")
    print("  CO4: Spatial Visualization (Maps)")
    print("  CO5: Interactive Price Analyzer (Power BI)")
    print("=" * 80)
    print("\n")
    
    # Create dataset
    print("Creating synthetic real estate dataset...")
    df = create_real_estate_dataset()
    
    # Save the dataset
    df.to_csv('output/real_estate_dataset.csv', index=False)
    print(f"[OK] Dataset created: {len(df)} properties")
    print("[OK] Saved: output/real_estate_dataset.csv\n")
    
    # Execute all analyses
    analyze_dataset_attributes(df)
    analyze_price_vs_area(df)
    create_property_treemap(df)
    create_spatial_visualization(df)
    create_powerbi_dataset(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nAll outputs saved in the 'output' directory:")
    print("  - CSV files: Dataset for analysis and Power BI")
    print("  - PNG files: Static visualizations")
    print("  - HTML files: Interactive maps and dashboards")
    print("  - TXT files: Power BI instructions")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()


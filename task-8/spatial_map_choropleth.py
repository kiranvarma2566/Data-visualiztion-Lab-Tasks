import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

# Create dataset with Tamil Nadu districts population data
data = {
    'District': ['Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 
                 'Erode', 'Kanchipuram', 'Kanyakumari', 'Karur', 'Krishnagiri',
                 'Madurai', 'Nagapattinam', 'Namakkal', 'Perambalur', 'Pudukkottai',
                 'Ramanathapuram', 'Salem', 'Sivaganga', 'Thanjavur', 'Theni',
                 'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli', 'Tiruppur',
                 'Tiruvallur', 'Tiruvannamalai', 'Tiruvarur', 'Vellore', 'Viluppuram', 'Virudhunagar'],
    '1991_Population': [5421985, 2441160, 1927821, 1764383, 1502851,
                        2109015, 2259316, 1638020, 816709, 1230142,
                        2482093, 1387791, 1160166, 448053, 1273880,
                        1111045, 2211453, 1032596, 2210285, 964969,
                        1435694, 1954047, 2375314, 1995756,
                        2358776, 2047969, 1067824, 2556244, 2519520, 1543806],
    '2001_Population': [6560942, 3021640, 2285395, 1960157, 1923014,
                        2251744, 2879916, 1676125, 936089, 1477010,
                        2989693, 1488684, 1496269, 516376, 1444542,
                        1214559, 3009636, 1144726, 2213216, 1093050,
                        1570977, 2359884, 2805223, 2479716,
                        2910756, 2158545, 1167793, 3477317, 2968015, 1795117],
    '2011_Population': [4681087, 3458045, 2605914, 1506843, 2159775,
                        2251744, 3998252, 1870374, 1064493, 1879809,
                        3038252, 1616450, 1726601, 565223, 1618725,
                        1353445, 3482056, 1339101, 2405890, 1245899,
                        1750176, 2722290, 3077233, 2479716,
                        3728104, 2464875, 1264277, 3936331, 3458873, 1942288],
    'Latitude': [13.08, 11.01, 11.75, 12.13, 10.36,
                 11.34, 12.83, 8.08, 10.96, 12.51,
                 9.92, 10.76, 11.22, 11.23, 10.38,
                 9.36, 11.66, 9.84, 10.78, 10.01,
                 8.76, 10.79, 8.71, 11.10,
                 13.13, 12.23, 10.77, 12.91, 11.94, 9.56],
    'Longitude': [80.27, 76.95, 79.77, 78.15, 77.98,
                  77.71, 79.70, 77.54, 78.07, 78.21,
                  78.11, 79.84, 78.16, 78.88, 78.82,
                  78.83, 78.14, 78.48, 79.13, 77.48,
                  78.13, 78.70, 77.75, 77.34,
                  80.09, 79.06, 79.63, 79.13, 79.49, 77.95]
}

df = pd.DataFrame(data)

# Calculate growth metrics
df['Growth_1991_2001'] = ((df['2001_Population'] - df['1991_Population']) / df['1991_Population']) * 100
df['Growth_2001_2011'] = ((df['2011_Population'] - df['2001_Population']) / df['2001_Population']) * 100
df['Total_Growth'] = ((df['2011_Population'] - df['1991_Population']) / df['1991_Population']) * 100
df['Population_Density_Index'] = df['2011_Population'] / 100000

# Tamil Nadu district boundaries (approximate polygons)
# These are simplified polygons for visualization purposes
district_boundaries = {
    'Chennai': [(80.15, 13.20), (80.30, 13.20), (80.30, 12.95), (80.15, 12.95)],
    'Tiruvallur': [(79.90, 13.40), (80.15, 13.40), (80.15, 12.95), (79.90, 12.95)],
    'Kanchipuram': [(79.60, 13.00), (79.95, 13.00), (79.95, 12.65), (79.60, 12.65)],
    'Vellore': [(78.85, 13.15), (79.40, 13.15), (79.40, 12.65), (78.85, 12.65)],
    'Tiruvannamalai': [(78.85, 12.65), (79.40, 12.65), (79.40, 12.05), (78.85, 12.05)],
    'Viluppuram': [(79.20, 12.35), (79.85, 12.35), (79.85, 11.65), (79.20, 11.65)],
    'Cuddalore': [(79.50, 12.05), (80.00, 12.05), (80.00, 11.40), (79.50, 11.40)],
    'Dharmapuri': [(77.85, 12.45), (78.60, 12.45), (78.60, 11.85), (77.85, 11.85)],
    'Krishnagiri': [(77.85, 12.85), (78.60, 12.85), (78.60, 12.25), (77.85, 12.25)],
    'Salem': [(77.90, 11.95), (78.50, 11.95), (78.50, 11.40), (77.90, 11.40)],
    'Namakkal': [(77.85, 11.50), (78.50, 11.50), (78.50, 11.05), (77.85, 11.05)],
    'Erode': [(77.40, 11.65), (77.95, 11.65), (77.95, 10.95), (77.40, 10.95)],
    'Coimbatore': [(76.70, 11.35), (77.25, 11.35), (77.25, 10.65), (76.70, 10.65)],
    'Tiruppur': [(77.10, 11.35), (77.55, 11.35), (77.55, 10.95), (77.10, 10.95)],
    'Karur': [(77.85, 11.15), (78.30, 11.15), (78.30, 10.75), (77.85, 10.75)],
    'Tiruchirappalli': [(78.45, 11.15), (78.95, 11.15), (78.95, 10.50), (78.45, 10.50)],
    'Perambalur': [(78.65, 11.45), (79.05, 11.45), (79.05, 11.05), (78.65, 11.05)],
    'Ariyalur': [(79.00, 11.35), (79.35, 11.35), (79.35, 11.00), (79.00, 11.00)],
    'Thanjavur': [(79.00, 11.10), (79.50, 11.10), (79.50, 10.50), (79.00, 10.50)],
    'Tiruvarur': [(79.35, 10.95), (79.85, 10.95), (79.85, 10.45), (79.35, 10.45)],
    'Nagapattinam': [(79.60, 11.00), (79.95, 11.00), (79.95, 10.40), (79.60, 10.40)],
    'Pudukkottai': [(78.70, 10.70), (79.20, 10.70), (79.20, 10.05), (78.70, 10.05)],
    'Dindigul': [(77.65, 10.70), (78.30, 10.70), (78.30, 10.05), (77.65, 10.05)],
    'Madurai': [(77.85, 10.25), (78.45, 10.25), (78.45, 9.65), (77.85, 9.65)],
    'Theni': [(77.20, 10.35), (77.80, 10.35), (77.80, 9.75), (77.20, 9.75)],
    'Sivaganga': [(78.30, 10.20), (79.00, 10.20), (79.00, 9.55), (78.30, 9.55)],
    'Ramanathapuram': [(78.60, 9.75), (79.30, 9.75), (79.30, 9.05), (78.60, 9.05)],
    'Virudhunagar': [(77.60, 9.90), (78.25, 9.90), (78.25, 9.25), (77.60, 9.25)],
    'Tirunelveli': [(77.40, 9.05), (77.95, 9.05), (77.95, 8.40), (77.40, 8.40)],
    'Thoothukudi': [(77.95, 9.15), (78.45, 9.15), (78.45, 8.45), (77.95, 8.45)],
    'Kanyakumari': [(77.20, 8.40), (77.70, 8.40), (77.70, 8.05), (77.20, 8.05)],
}

# Create the visualization
fig = plt.figure(figsize=(24, 14))
fig.suptitle('Spatial and Geospatial Analysis - Tamil Nadu Districts\nPopulation Statistics and Growth Analysis (1991-2011)', 
             fontsize=24, fontweight='bold', y=0.98)

# Color maps for different metrics
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# 1. Population Distribution Map (2011) - Main Choropleth
ax1 = plt.subplot(2, 3, 1)
ax1.set_xlim(76.5, 80.5)
ax1.set_ylim(8.0, 13.5)
ax1.set_aspect('equal')
ax1.set_facecolor('#E8F4F8')

# Create color mapping for population
norm_pop = Normalize(vmin=df['2011_Population'].min(), vmax=df['2011_Population'].max())
cmap_pop = plt.cm.YlOrRd

patches = []
colors = []
for idx, row in df.iterrows():
    district = row['District']
    if district in district_boundaries:
        coords = district_boundaries[district]
        polygon = Polygon(coords, closed=True)
        patches.append(polygon)
        colors.append(cmap_pop(norm_pop(row['2011_Population'])))

p = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=1.5)
ax1.add_collection(p)

# Add district labels for major districts
for idx, row in df.nlargest(10, '2011_Population').iterrows():
    ax1.text(row['Longitude'], row['Latitude'], row['District'], 
            fontsize=8, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax1.set_title('Population Distribution by District (2011)\nChoropleth Map', 
              fontsize=14, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add colorbar
sm = ScalarMappable(cmap=cmap_pop, norm=norm_pop)
sm.set_array([])
cbar1 = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Population (2011)', fontsize=11, fontweight='bold')

# 2. Growth Rate Choropleth Map (1991-2011)
ax2 = plt.subplot(2, 3, 2)
ax2.set_xlim(76.5, 80.5)
ax2.set_ylim(8.0, 13.5)
ax2.set_aspect('equal')
ax2.set_facecolor('#F5F5F5')

norm_growth = Normalize(vmin=df['Total_Growth'].min(), vmax=df['Total_Growth'].max())
cmap_growth = plt.cm.RdYlGn

patches2 = []
colors2 = []
for idx, row in df.iterrows():
    district = row['District']
    if district in district_boundaries:
        coords = district_boundaries[district]
        polygon = Polygon(coords, closed=True)
        patches2.append(polygon)
        colors2.append(cmap_growth(norm_growth(row['Total_Growth'])))

p2 = PatchCollection(patches2, facecolors=colors2, edgecolors='black', linewidths=1.5)
ax2.add_collection(p2)

# Label high growth districts
for idx, row in df.nlargest(8, 'Total_Growth').iterrows():
    ax2.text(row['Longitude'], row['Latitude'], f"{row['District']}\n+{row['Total_Growth']:.1f}%", 
            fontsize=7, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

ax2.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax2.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax2.set_title('Population Growth Rate (1991-2011)\nPercentage Change', 
              fontsize=14, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--')

sm2 = ScalarMappable(cmap=cmap_growth, norm=norm_growth)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Growth Rate (%)', fontsize=11, fontweight='bold')

# 3. Top Districts by Population
ax3 = plt.subplot(2, 3, 3)
top_districts = df.nlargest(12, '2011_Population')
colors3 = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_districts)))
bars = ax3.barh(top_districts['District'], top_districts['2011_Population'], 
                color=colors3, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Population (2011)', fontsize=12, fontweight='bold')
ax3.set_ylabel('District', fontsize=12, fontweight='bold')
ax3.set_title('Top 12 Districts by Population', fontsize=14, fontweight='bold', pad=10)
ax3.grid(axis='x', alpha=0.3)
for bar in bars:
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, 
            f'{int(width/1000)}K', 
            ha='left', va='center', fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

# 4. Temporal Trends
ax4 = plt.subplot(2, 3, 4)
years = ['1991', '2001', '2011']
top_districts_trend = df.nlargest(8, '2011_Population')

for idx, row in top_districts_trend.iterrows():
    populations = [row['1991_Population'], row['2001_Population'], row['2011_Population']]
    ax4.plot(years, populations, marker='o', linewidth=2.5, markersize=10, 
            label=row['District'], alpha=0.8)

ax4.set_xlabel('Census Year', fontsize=12, fontweight='bold')
ax4.set_ylabel('Population', fontsize=12, fontweight='bold')
ax4.set_title('Population Trends - Top 8 Districts', fontsize=14, fontweight='bold', pad=10)
ax4.legend(loc='upper left', fontsize=10, ncol=2)
ax4.grid(True, alpha=0.4, linestyle='--')
ax4.ticklabel_format(style='plain', axis='y')

# 5. Population Density Heatmap Style
ax5 = plt.subplot(2, 3, 5)
ax5.set_xlim(76.5, 80.5)
ax5.set_ylim(8.0, 13.5)
ax5.set_aspect('equal')
ax5.set_facecolor('#F0F0F0')

# Bubble size map with district boundaries
norm_density = Normalize(vmin=df['Population_Density_Index'].min(), 
                        vmax=df['Population_Density_Index'].max())
cmap_density = plt.cm.plasma

patches5 = []
colors5 = []
for idx, row in df.iterrows():
    district = row['District']
    if district in district_boundaries:
        coords = district_boundaries[district]
        polygon = Polygon(coords, closed=True)
        patches5.append(polygon)
        colors5.append(cmap_density(norm_density(row['Population_Density_Index'])))

p5 = PatchCollection(patches5, facecolors=colors5, edgecolors='black', linewidths=1.5, alpha=0.7)
ax5.add_collection(p5)

# Add bubble overlay
scatter5 = ax5.scatter(df['Longitude'], df['Latitude'], 
                      s=df['2011_Population']/5000,
                      c='white', alpha=0.3, edgecolors='black', linewidth=2)

ax5.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax5.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax5.set_title('Population Density Index\nCombined Choropleth & Bubble Map', 
              fontsize=14, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, linestyle='--')

sm5 = ScalarMappable(cmap=cmap_density, norm=norm_density)
sm5.set_array([])
cbar5 = plt.colorbar(sm5, ax=ax5, fraction=0.046, pad=0.04)
cbar5.set_label('Density Index', fontsize=11, fontweight='bold')

# 6. Growth Distribution Categories
ax6 = plt.subplot(2, 3, 6)
ax6.set_xlim(76.5, 80.5)
ax6.set_ylim(8.0, 13.5)
ax6.set_aspect('equal')
ax6.set_facecolor('#FFFEF0')

# Categorize growth
def get_growth_category(growth):
    if growth > 50:
        return 'High Growth (>50%)', '#2ecc71'
    elif growth > 20:
        return 'Moderate Growth (20-50%)', '#f39c12'
    elif growth > 0:
        return 'Low Growth (0-20%)', '#3498db'
    else:
        return 'Negative Growth', '#e74c3c'

patches6 = []
colors6 = []
categories = []

for idx, row in df.iterrows():
    district = row['District']
    if district in district_boundaries:
        coords = district_boundaries[district]
        polygon = Polygon(coords, closed=True)
        patches6.append(polygon)
        category, color = get_growth_category(row['Total_Growth'])
        colors6.append(color)
        if category not in categories:
            categories.append(category)

p6 = PatchCollection(patches6, facecolors=colors6, edgecolors='black', linewidths=1.5)
ax6.add_collection(p6)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71', markersize=12, 
           label='High Growth (>50%)', markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#f39c12', markersize=12, 
           label='Moderate Growth (20-50%)', markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db', markersize=12, 
           label='Low Growth (0-20%)', markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', markersize=12, 
           label='Negative Growth', markeredgecolor='black')
]
ax6.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

ax6.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax6.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax6.set_title('Growth Rate Categories (1991-2011)\nCategorical Classification', 
              fontsize=14, fontweight='bold', pad=10)
ax6.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
output_filename = 'tamil_nadu_choropleth_map.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')

print("=" * 90)
print("TAMIL NADU DISTRICTS - SPATIAL GEOSPATIAL ANALYSIS WITH CHOROPLETH MAPS")
print("=" * 90)
print("\nDataset Overview:")
print(df[['District', '1991_Population', '2001_Population', '2011_Population', 'Total_Growth']].head(10))
print("\n" + "=" * 90)
print("\nKEY INSIGHTS:")
print("=" * 90)
print(f"Total districts analyzed: {len(df)}")
print(f"Most populous district (2011): {df.loc[df['2011_Population'].idxmax(), 'District']} ({df['2011_Population'].max():,})")
print(f"Least populous district (2011): {df.loc[df['2011_Population'].idxmin(), 'District']} ({df['2011_Population'].min():,})")
print(f"Highest growth rate: {df.loc[df['Total_Growth'].idxmax(), 'District']} ({df['Total_Growth'].max():.2f}%)")
print(f"Lowest growth rate: {df.loc[df['Total_Growth'].idxmin(), 'District']} ({df['Total_Growth'].min():.2f}%)")
print(f"Average population growth (1991-2011): {df['Total_Growth'].mean():.2f}%")
print(f"Total Tamil Nadu population (2011): {df['2011_Population'].sum():,}")
print("\n" + "=" * 90)
print(f"Output saved as: {output_filename}")
print("=" * 90)

plt.show()



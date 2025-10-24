import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Create sample dataset based on Tamil Nadu cities and towns
# In practice, you would load this from the Kaggle dataset
data = {
    'Name': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 
             'Tirunelveli', 'Tiruppur', 'Erode', 'Vellore', 'Thoothukudi',
             'Dindigul', 'Thanjavur', 'Ranipet', 'Sivakasi', 'Karur',
             'Kanchipuram', 'Kumarapalayam', 'Neyveli', 'Cuddalore', 'Kumbakonam',
             'Avadi', 'Tambaram', 'Hosur', 'Nagercoil', 'Pollachi'],
    'Status': ['City', 'City', 'City', 'City', 'City', 
               'City', 'City', 'City', 'City', 'City',
               'City', 'City', 'Town', 'Town', 'City',
               'City', 'Town', 'Town', 'City', 'City',
               'City', 'City', 'City', 'City', 'Town'],
    'District': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem',
                 'Tirunelveli', 'Tiruppur', 'Erode', 'Vellore', 'Thoothukudi',
                 'Dindigul', 'Thanjavur', 'Ranipet', 'Virudhunagar', 'Karur',
                 'Kanchipuram', 'Tiruppur', 'Cuddalore', 'Cuddalore', 'Thanjavur',
                 'Tiruvallur', 'Chengalpattu', 'Krishnagiri', 'Kanyakumari', 'Coimbatore'],
    '1991-03-01': [3841396, 816321, 940989, 711548, 366712,
                   395622, 211417, 151184, 177230, 284540,
                   182477, 215314, 52000, 68586, 76915,
                   145143, 48500, 62000, 158877, 139264,
                   184450, 124690, 53091, 189651, 72100],
    '2001-03-01': [4343645, 930882, 1017524, 752066, 693236,
                   411831, 344543, 198131, 177081, 347881,
                   207327, 215725, 59000, 71170, 76915,
                   164384, 51000, 75000, 158336, 140156,
                   226564, 142000, 68986, 208179, 76500],
    '2011-03-01': [4681087, 1050721, 1017865, 847387, 831038,
                   474838, 444543, 215314, 423425, 407402,
                   207327, 222943, 66634, 77799, 106712,
                   164384, 58413, 85000, 173636, 140136,
                   345996, 159909, 109510, 224849, 90180],
    # Geographical coordinates (latitude, longitude)
    'Latitude': [13.0827, 11.0168, 9.9252, 10.7905, 11.6643,
                 8.7139, 11.1085, 11.3410, 12.9165, 8.7642,
                 10.3673, 10.7870, 12.9249, 9.4515, 10.9601,
                 12.8346, 11.1230, 11.6053, 11.7480, 10.9617,
                 13.1067, 12.9249, 12.7409, 8.1790, 10.6580],
    'Longitude': [80.2707, 76.9558, 78.1198, 78.7047, 78.1460,
                  77.7567, 77.3411, 77.7172, 79.1325, 78.1348,
                  77.9803, 79.1378, 79.3308, 77.8081, 78.0766,
                  79.7036, 77.4000, 79.8600, 79.7714, 79.3881,
                  80.1109, 80.1270, 77.8253, 77.4305, 77.0066]
}

df = pd.DataFrame(data)

# Calculate population growth and other metrics
df['Growth_1991_2001'] = ((df['2001-03-01'] - df['1991-03-01']) / df['1991-03-01']) * 100
df['Growth_2001_2011'] = ((df['2011-03-01'] - df['2001-03-01']) / df['2001-03-01']) * 100
df['Total_Growth'] = ((df['2011-03-01'] - df['1991-03-01']) / df['1991-03-01']) * 100
df['Population_Density_Index'] = df['2011-03-01'] / 1000  # Simplified index

print("=" * 80)
print("TAMIL NADU CITIES AND TOWNS - SPATIAL GEOSPATIAL ANALYSIS")
print("=" * 80)
print("\nDataset Overview:")
print(df.head(10))
print("\n" + "=" * 80)
print("\nPopulation Statistics Summary:")
print(df[['1991-03-01', '2001-03-01', '2011-03-01']].describe())
print("\n" + "=" * 80)
print("\nTop 5 Cities by 2011 Population:")
print(df.nlargest(5, '2011-03-01')[['Name', 'District', '2011-03-01']])
print("\n" + "=" * 80)
print("\nTop 5 Cities by Total Growth (1991-2011):")
print(df.nlargest(5, 'Total_Growth')[['Name', 'Total_Growth']])
print("=" * 80)

# Create comprehensive spatial visualization
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Spatial and Geospatial Analysis of Tamil Nadu Cities and Towns\nPopulation Statistics (1991-2011)', 
             fontsize=20, fontweight='bold', y=0.98)

# Define Tamil Nadu boundaries (approximate)
TN_LAT_MIN, TN_LAT_MAX = 8.0, 13.5
TN_LON_MIN, TN_LON_MAX = 76.5, 80.5

# 1. Geographical Map with Population (2011 Census)
ax1 = plt.subplot(2, 3, 1)
# Create background map
ax1.set_xlim(TN_LON_MIN, TN_LON_MAX)
ax1.set_ylim(TN_LAT_MIN, TN_LAT_MAX)
ax1.set_facecolor('#E8F4F8')
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot cities with size proportional to 2011 population
scatter1 = ax1.scatter(df['Longitude'], df['Latitude'], 
                      s=df['2011-03-01']/3000,  # Scale for visibility
                      c=df['2011-03-01'], 
                      cmap='YlOrRd', 
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=1.5)

# Annotate major cities
for idx, row in df.nlargest(8, '2011-03-01').iterrows():
    ax1.annotate(row['Name'], 
                xy=(row['Longitude'], row['Latitude']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax1.set_xlabel('Longitude', fontsize=10, fontweight='bold')
ax1.set_ylabel('Latitude', fontsize=10, fontweight='bold')
ax1.set_title('Geographic Distribution - 2011 Population\n(Bubble size = Population)', 
              fontsize=12, fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Population', fontsize=9)

# 2. Population Growth Rate Map (1991-2011)
ax2 = plt.subplot(2, 3, 2)
ax2.set_xlim(TN_LON_MIN, TN_LON_MAX)
ax2.set_ylim(TN_LAT_MIN, TN_LAT_MAX)
ax2.set_facecolor('#F0F0F0')
ax2.grid(True, alpha=0.3, linestyle='--')

scatter2 = ax2.scatter(df['Longitude'], df['Latitude'], 
                      s=300,
                      c=df['Total_Growth'], 
                      cmap='RdYlGn', 
                      alpha=0.7,
                      edgecolors='black',
                      linewidth=1.5,
                      vmin=df['Total_Growth'].min(),
                      vmax=df['Total_Growth'].max())

# Annotate high growth cities
for idx, row in df.nlargest(5, 'Total_Growth').iterrows():
    ax2.annotate(f"{row['Name']}\n+{row['Total_Growth']:.1f}%", 
                xy=(row['Longitude'], row['Latitude']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

ax2.set_xlabel('Longitude', fontsize=10, fontweight='bold')
ax2.set_ylabel('Latitude', fontsize=10, fontweight='bold')
ax2.set_title('Population Growth Rate (1991-2011)\n(% Change)', 
              fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Growth %', fontsize=9)

# 3. District-wise Analysis
ax3 = plt.subplot(2, 3, 3)
district_pop = df.groupby('District')['2011-03-01'].sum().sort_values(ascending=False).head(10)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(district_pop)))
bars = ax3.barh(district_pop.index, district_pop.values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Total Population (2011)', fontsize=10, fontweight='bold')
ax3.set_ylabel('District', fontsize=10, fontweight='bold')
ax3.set_title('Top 10 Districts by Population (2011)', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, 
            f'{int(width):,}', 
            ha='left', va='center', fontsize=8, fontweight='bold')

# 4. Temporal Population Trends
ax4 = plt.subplot(2, 3, 4)
years = ['1991', '2001', '2011']
top_cities = df.nlargest(10, '2011-03-01')

for idx, row in top_cities.iterrows():
    populations = [row['1991-03-01'], row['2001-03-01'], row['2011-03-01']]
    ax4.plot(years, populations, marker='o', linewidth=2, markersize=8, label=row['Name'], alpha=0.7)

ax4.set_xlabel('Census Year', fontsize=10, fontweight='bold')
ax4.set_ylabel('Population', fontsize=10, fontweight='bold')
ax4.set_title('Population Trends - Top 10 Cities', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8, ncol=2)
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(style='plain', axis='y')

# 5. Spatial Clustering by City/Town Status
ax5 = plt.subplot(2, 3, 5)
ax5.set_xlim(TN_LON_MIN, TN_LON_MAX)
ax5.set_ylim(TN_LAT_MIN, TN_LAT_MAX)
ax5.set_facecolor('#FFF8DC')
ax5.grid(True, alpha=0.3, linestyle='--')

for status in df['Status'].unique():
    subset = df[df['Status'] == status]
    marker = 'o' if status == 'City' else '^'
    color = 'red' if status == 'City' else 'blue'
    ax5.scatter(subset['Longitude'], subset['Latitude'], 
               s=200, marker=marker, c=color, alpha=0.6, 
               edgecolors='black', linewidth=1.5, label=status)

ax5.set_xlabel('Longitude', fontsize=10, fontweight='bold')
ax5.set_ylabel('Latitude', fontsize=10, fontweight='bold')
ax5.set_title('Spatial Distribution by Status\n(Cities vs Towns)', 
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)

# 6. Growth Rate Distribution
ax6 = plt.subplot(2, 3, 6)
ax6.set_xlim(TN_LON_MIN, TN_LON_MAX)
ax6.set_ylim(TN_LAT_MIN, TN_LAT_MAX)
ax6.set_facecolor('#E8F8E8')
ax6.grid(True, alpha=0.3, linestyle='--')

# Create arrows showing growth direction and magnitude
for idx, row in df.iterrows():
    growth_2001_2011 = row['Growth_2001_2011']
    # Arrow length and color based on growth rate
    arrow_length = abs(growth_2001_2011) / 500  # Scale factor
    color = 'green' if growth_2001_2011 > 10 else 'orange' if growth_2001_2011 > 0 else 'red'
    
    ax6.scatter(row['Longitude'], row['Latitude'], 
               s=150, c=color, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Draw arrow pointing up for growth
    if abs(growth_2001_2011) > 5:
        ax6.arrow(row['Longitude'], row['Latitude'], 
                 0, arrow_length, 
                 head_width=0.08, head_length=0.05, 
                 fc=color, ec='black', alpha=0.7, linewidth=1)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, 
           label='High Growth (>10%)', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, 
           label='Moderate Growth (0-10%)', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
           label='Low/Negative Growth', markeredgecolor='black')
]
ax6.legend(handles=legend_elements, loc='upper right', fontsize=8)

ax6.set_xlabel('Longitude', fontsize=10, fontweight='bold')
ax6.set_ylabel('Latitude', fontsize=10, fontweight='bold')
ax6.set_title('Growth Rate Distribution (2001-2011)\nArrow Height = Growth Magnitude', 
              fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
output_filename = 'tamil_nadu_spatial_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nVisualization saved as: {output_filename}")
print("\n" + "=" * 80)
print("KEY INSIGHTS FROM SPATIAL GEOSPATIAL ANALYSIS:")
print("=" * 80)
print(f"1. Total cities/towns analyzed: {len(df)}")
print(f"2. Cities: {len(df[df['Status']=='City'])}, Towns: {len(df[df['Status']=='Town'])}")
print(f"3. Most populous city (2011): {df.loc[df['2011-03-01'].idxmax(), 'Name']} ({df['2011-03-01'].max():,})")
print(f"4. Highest growth rate: {df.loc[df['Total_Growth'].idxmax(), 'Name']} ({df['Total_Growth'].max():.2f}%)")
print(f"5. Geographic spread: Latitude {TN_LAT_MIN}° to {TN_LAT_MAX}°, Longitude {TN_LON_MIN}° to {TN_LON_MAX}°")
print(f"6. Average population growth (1991-2011): {df['Total_Growth'].mean():.2f}%")
print(f"7. Total Tamil Nadu urban population (2011): {df['2011-03-01'].sum():,}")
print("=" * 80)

plt.show()


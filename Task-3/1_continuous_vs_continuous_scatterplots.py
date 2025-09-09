import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load and prepare data
    df = pd.read_csv('customer_summary_report.csv')
    df['Call Start Time'] = pd.to_datetime(df['Call Start Time'])
    df['Call Start Hour'] = df['Call Start Time'].dt.hour
    df['Call Start Minute'] = df['Call Start Time'].dt.minute
    df['Duration_Minutes'] = df['Duration (seconds)'] / 60
    df['Tower_Number'] = df['Tower ID'].str.extract(r'(\d+)').astype(int)
    df['Time_Since_Start'] = (df['Call Start Time'] - df['Call Start Time'].min()).dt.total_seconds() / 3600
    
    print("="*60)
    print("CONTINUOUS vs CONTINUOUS BIVARIATE ANALYSIS")
    print(f"Dataset: {df.shape[0]} records, {df.shape[1]} columns")
    
    # Define variable pairs
    pairs = [
        ('Duration (seconds)', 'Call Start Hour', 'Call Duration vs Start Hour'),
        ('Duration (seconds)', 'Tower_Number', 'Call Duration vs Tower Number'),
        ('Call Start Hour', 'Call Start Minute', 'Call Start Hour vs Start Minute'),
        ('Duration_Minutes', 'Time_Since_Start', 'Call Duration vs Time Since Start'),
        ('Duration (seconds)', 'Time_Since_Start', 'Call Duration vs Elapsed Time'),
        ('Tower_Number', 'Call Start Hour', 'Tower Number vs Call Start Hour')
    ]
    
    # Create scatterplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Continuous vs Continuous Analysis - Scatterplots with Fit Lines', fontsize=16, fontweight='bold')
    
    for i, (x_var, y_var, title) in enumerate(pairs):
        ax = axes[i//3, i%3]
        x, y = df[x_var].dropna(), df[y_var].dropna()
        min_len = min(len(x), len(y))
        x, y = x.iloc[:min_len], y.iloc[:min_len]
        
        ax.scatter(x, y, alpha=0.6, color='steelblue', s=50, edgecolors='white', linewidth=0.5)
        
        if len(x) > 1:
            corr = np.corrcoef(x, y)[0, 1] if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0
            
            # Linear fit
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear (r={corr:.3f})')
            
            # Polynomial fit
            try:
                z2 = np.polyfit(x, y, 2)
                ax.plot(x_line, np.poly1d(z2)(x_line), "g-", alpha=0.8, linewidth=2, label='Polynomial')
            except: pass
        
        ax.set_xlabel(x_var, fontsize=10)
        ax.set_ylabel(y_var, fontsize=10)
        ax.set_title(f'{title}\nr = {corr:.3f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('continuous_vs_continuous_scatterplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    continuous_vars = ['Duration (seconds)', 'Call Start Hour', 'Call Start Minute', 'Duration_Minutes', 'Tower_Number', 'Time_Since_Start']
    corr_matrix = df[continuous_vars].corr()
    print("Correlation Matrix:")
    print(corr_matrix.round(3))
    
    print("\nDetailed Pair Analysis:")
    for x_var, y_var, title in pairs:
        x, y = df[x_var].dropna(), df[y_var].dropna()
        if len(x) > 1:
            pearson_corr, pearson_p = stats.pearsonr(x, y)
            spearman_corr, spearman_p = stats.spearmanr(x, y)
            print(f"\n{title}:")
            print(f"  Pearson: {pearson_corr:.4f} (p={pearson_p:.4f})")
            print(f"  Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")
            print(f"  → {'Significant' if pearson_p < 0.05 else 'Not significant'}")
    
    # Regression analysis
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)
    
    x_var, y_var = 'Time_Since_Start', 'Duration (seconds)'
    x, y = df[[x_var]].dropna(), df[y_var].dropna()
    min_len = min(len(x), len(y))
    x, y = x.iloc[:min_len], y.iloc[:min_len]
    
    if len(x) > 5:
        # Linear and polynomial regression
        linear_reg = LinearRegression()
        linear_reg.fit(x, y)
        poly_reg = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly_reg.fit(x, y)
        
        y_pred_linear = linear_reg.predict(x)
        y_pred_poly = poly_reg.predict(x)
        
        r2_linear = r2_score(y, y_pred_linear)
        r2_poly = r2_score(y, y_pred_poly)
        
        print(f"Linear R²: {r2_linear:.4f}")
        print(f"Polynomial R²: {r2_poly:.4f}")
        print(f"Linear equation: y = {linear_reg.coef_[0]:.4f}x + {linear_reg.intercept_:.4f}")
        
        # Regression plots
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, alpha=0.6, color='blue')
        plt.plot(x, y_pred_linear, color='red', linewidth=2, label=f'Linear (R²={r2_linear:.3f})')
        plt.plot(x, y_pred_poly, color='green', linewidth=2, label=f'Polynomial (R²={r2_poly:.3f})')
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title('Regression Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.scatter(y_pred_linear, y - y_pred_linear, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Linear Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.scatter(y_pred_poly, y - y_pred_poly, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Polynomial Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("Generated: continuous_vs_continuous_scatterplots.png, regression_analysis.png")
    print("="*60)

if __name__ == "__main__":
    main()
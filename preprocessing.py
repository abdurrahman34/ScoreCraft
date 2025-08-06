import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define file paths directly
input_file = 'application_train_with_features.csv'

# Load the data
print("Loading data...")
try:
    df = pd.read_csv(input_file)
    print(f"Data loaded successfully with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File {input_file} not found. Make sure to run Create_New_File.py first.")
    exit(1)


# Basic information about the dataset
print("\nBasic information about the dataset:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Separate features into numerical and categorical
print("\nSeparating features into numerical and categorical...")

# First, identify the target variable
target = 'TARGET' if 'TARGET' in df.columns else None
if target:
    print(f"Target variable identified: {target}")
    y = df[target]
    X = df.drop(columns=[target])
else:
    print("Warning: TARGET column not found. Using all columns as features.")
    X = df.copy()

# Identify numerical and categorical features
num_features = []
cat_features = []

for col in X.columns:
    # Check if the column is numerical
    if X[col].dtype in ['int64', 'float64']:
        num_features.append(col)
    else:
        cat_features.append(col)

# Check if TARGET column exists in the original dataframe
if 'TARGET' in df.columns:
    # Extract target variable before it was potentially removed in previous steps
    y = df['TARGET']
    
    # Make sure we have the same number of samples in X and y
    # This is needed because we removed some rows during outlier removal
    common_index = X.index.intersection(df.index)
    X_subset = X.loc[common_index]
    y = df.loc[common_index, 'TARGET']
    
    print(f"\nUsing {len(X_subset)} samples for feature importance analysis")
    
    # Select features for analysis (exclude ID columns and TARGET)
    exclude_cols = ['Unnamed: 0', 'SK_ID_CURR', 'TARGET', 'ID']
    features = [col for col in X_subset.columns if col not in exclude_cols]
    X_subset = X_subset[features]

print(f"\nNumerical features ({len(num_features)}): {num_features[:5]}...")
print(f"Categorical features ({len(cat_features)}): {cat_features[:5]}...")

# Preprocessing steps
print("\nStarting preprocessing...")

# 1. Handle missing values
print("\n1. Handling missing values...")

# Specifically handle OWN_CAR_AGE missing values by replacing them with -1
if 'OWN_CAR_AGE' in X.columns:
    print(f"\nHandling missing values in OWN_CAR_AGE column...")
    print(f"Missing values in OWN_CAR_AGE before: {X['OWN_CAR_AGE'].isnull().sum()}")
    
    # Replace missing values with -1 (indicating no car or missing information)
    X['OWN_CAR_AGE'] = X['OWN_CAR_AGE'].fillna(-1)
    
    # Make sure OWN_CAR_AGE is treated as a numerical feature
    if 'OWN_CAR_AGE' in cat_features:
        cat_features.remove('OWN_CAR_AGE')
    
    if 'OWN_CAR_AGE' not in num_features:
        num_features.append('OWN_CAR_AGE')
        print("OWN_CAR_AGE set as a numerical feature with -1 for missing values")
    
    print(f"Missing values in OWN_CAR_AGE after: {X['OWN_CAR_AGE'].isnull().sum()}")



# Handle missing values in OCCUPATION_TYPE by replacing them with 'None'
if 'OCCUPATION_TYPE' in X.columns:
    print(f"\nHandling missing values in OCCUPATION_TYPE column...")
    print(f"Missing values in OCCUPATION_TYPE before: {X['OCCUPATION_TYPE'].isnull().sum()}")
    
    # Replace missing values with 'None'
    X['OCCUPATION_TYPE'] = X['OCCUPATION_TYPE'].fillna('None')
    
    print(f"Missing values in OCCUPATION_TYPE after: {X['OCCUPATION_TYPE'].isnull().sum()}")

# Handle missing values in AMT_GOODS_PRICE by replacing them with 0
if 'AMT_GOODS_PRICE' in X.columns:
    print(f"\nHandling missing values in AMT_GOODS_PRICE column...")
    print(f"Missing values in AMT_GOODS_PRICE before: {X['AMT_GOODS_PRICE'].isnull().sum()}")
    
    # Replace missing values with 0
    X['AMT_GOODS_PRICE'] = X['AMT_GOODS_PRICE'].fillna(0)
    
    print(f"Missing values in AMT_GOODS_PRICE after: {X['AMT_GOODS_PRICE'].isnull().sum()}")

# Check remaining missing values before removing rows
print("\nChecking remaining missing values before removing rows:")
num_missing = X[num_features].isnull().sum()
cat_missing = X[cat_features].isnull().sum()
print(f"Numerical features with missing values:\n{num_missing[num_missing > 0]}")
print(f"Categorical features with missing values:\n{cat_missing[cat_missing > 0]}")

# Remove all remaining rows with missing values
print(f"\nRemoving all remaining rows with missing values...")
print(f"Shape before removing rows with missing values: {X.shape}")

# Store the indices of rows to be removed
rows_with_missing = X.isnull().any(axis=1)
print(f"Number of rows with missing values: {rows_with_missing.sum()}")

# Remove rows with missing values from both X and y if target exists
X = X.dropna()
if target:
    # Make sure y is aligned with X after dropping rows
    y = y.loc[X.index]

print(f"Shape after removing rows with missing values: {X.shape}")

# Verify no missing values remain
remaining_missing = X.isnull().sum().sum()
print(f"\nRemaining missing values after removal: {remaining_missing}")

# Display unique values for each categorical attribute
print("\n" + "="*50)
print("UNIQUE VALUES FOR EACH CATEGORICAL ATTRIBUTE")
print("="*50)

for col in cat_features:
    unique_values = X[col].unique()
    # Calculate both counts and percentages
    value_counts = X[col].value_counts()
    value_percentages = X[col].value_counts(normalize=True) * 100
    
    print(f"\n{col} - {len(unique_values)} unique values:")
    
    # Display all values and their percentages
    for value, count in value_counts.items():
        percentage = value_percentages[value]
        print(f"{value}: {count} ({percentage:.2f}%)")

# Özel olarak DAYS_EMPLOYED değişkenini inceleyelim
print("\n" + "="*50)
print("DAYS_EMPLOYED İNCELEMESİ")
print("="*50)

# Pozitif ve negatif değerlerin sayısını bulalım
days_employed_positive = (X['DAYS_EMPLOYED'] > 0).sum()
days_employed_negative = (X['DAYS_EMPLOYED'] < 0).sum()
days_employed_zero = (X['DAYS_EMPLOYED'] == 0).sum()

print(f"DAYS_EMPLOYED pozitif değer sayısı: {days_employed_positive} ({days_employed_positive/len(X)*100:.2f}%)")
print(f"DAYS_EMPLOYED negatif değer sayısı: {days_employed_negative} ({days_employed_negative/len(X)*100:.2f}%)")
print(f"DAYS_EMPLOYED sıfır değer sayısı: {days_employed_zero} ({days_employed_zero/len(X)*100:.2f}%)")

# En büyük değeri kontrol edelim
max_days_employed = X['DAYS_EMPLOYED'].max()
print(f"\nEn büyük DAYS_EMPLOYED değeri: {max_days_employed}")

# 365243 değerini içeren kayıtların sayısını bulalım
days_employed_365243 = (X['DAYS_EMPLOYED'] == 365243).sum()
print(f"DAYS_EMPLOYED = 365243 olan kayıt sayısı: {days_employed_365243} ({days_employed_365243/len(X)*100:.2f}%)")

# 365243 değerini içermeyen kayıtların istatistiklerini hesaplayalım
days_employed_normal = X[X['DAYS_EMPLOYED'] != 365243]['DAYS_EMPLOYED']
print("\nDAYS_EMPLOYED (365243 değeri hariç) istatistikleri:")
print(f"Ortalama: {days_employed_normal.mean():.2f}")
print(f"Varyans: {days_employed_normal.var():.2f}")
print(f"Standart Sapma: {days_employed_normal.std():.2f}")
print(f"Minimum: {days_employed_normal.min():.2f}")
print(f"25%: {days_employed_normal.quantile(0.25):.2f}")
print(f"Medyan: {days_employed_normal.median():.2f}")
print(f"75%: {days_employed_normal.quantile(0.75):.2f}")
print(f"Maksimum: {days_employed_normal.max():.2f}")

# Handle DAYS_EMPLOYED with value 365243 (likely representing emekli)
if 'DAYS_EMPLOYED' in X.columns:
    print(f"\nHandling anomalies in DAYS_EMPLOYED column...")
    anomaly_count = (X['DAYS_EMPLOYED'] == 365243).sum()
    print(f"Number of records with DAYS_EMPLOYED = 365243: {anomaly_count} ({anomaly_count/len(X)*100:.2f}%)")
    
    # Replace 365243 values with -50 (indicating unemployed or special case)
    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, -50)
    
    print(f"DAYS_EMPLOYED = 365243 values have been replaced with -50")
    print(f"Number of records with DAYS_EMPLOYED = -50 now: {(X['DAYS_EMPLOYED'] == -50).sum()}")
    
    # Update EMPLOYMENT_YEARS based on the new DAYS_EMPLOYED values
    if 'EMPLOYMENT_YEARS' in X.columns:
        print(f"\nUpdating EMPLOYMENT_YEARS based on new DAYS_EMPLOYED values...")
        # Recalculate EMPLOYMENT_YEARS using the same formula as in Create_New_File.py
        X['EMPLOYMENT_YEARS'] = (abs(X['DAYS_EMPLOYED']) / 365.25).round(1)
        
        # Show statistics for the updated EMPLOYMENT_YEARS
        print(f"EMPLOYMENT_YEARS statistics after update:")
        print(f"Mean: {X['EMPLOYMENT_YEARS'].mean():.2f}")
        print(f"Min: {X['EMPLOYMENT_YEARS'].min():.2f}")
        print(f"Max: {X['EMPLOYMENT_YEARS'].max():.2f}")
        print(f"Number of records with EMPLOYMENT_YEARS = 0.1: {(X['EMPLOYMENT_YEARS'] == 0.1).sum()} (converted from DAYS_EMPLOYED = -50)")




# Calculate and display mean and variance for numerical attributes
print("\n" + "="*50)
print("MEAN AND VARIANCE FOR EACH NUMERICAL ATTRIBUTE")
print("="*50)

# Create a DataFrame to store statistics
stats_df = pd.DataFrame(index=num_features)

# Exclude Unnamed and SK_ID_CURR columns from statistics calculation
columns_to_exclude = ['Unnamed: 0', 'SK_ID_CURR']
num_features_for_stats = [col for col in num_features if col not in columns_to_exclude]

# Create a new DataFrame only for features we want to show statistics for
stats_df = pd.DataFrame(index=num_features_for_stats)

# Calculate statistics for each numerical feature (excluding Unnamed and SK_ID_CURR)
for col in num_features_for_stats:
    try:
        stats_df.loc[col, 'Mean'] = X[col].mean()
        stats_df.loc[col, 'Variance'] = X[col].var()
        stats_df.loc[col, 'Std Dev'] = X[col].std()
        stats_df.loc[col, 'Min'] = X[col].min()
        stats_df.loc[col, 'Max'] = X[col].max()
    except:
        stats_df.loc[col, 'Mean'] = 'N/A'
        stats_df.loc[col, 'Variance'] = 'N/A'
        stats_df.loc[col, 'Std Dev'] = 'N/A'
        stats_df.loc[col, 'Min'] = 'N/A'
        stats_df.loc[col, 'Max'] = 'N/A'

# Display the statistics
print("\nStatistics for numerical features:")
print(stats_df)

# Display more detailed information for important numerical features
print("\nDetailed statistics for key numerical features:")
key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
key_features = [f for f in key_features if f in num_features]




# Show detailed statistics only for features we want (excluding ID columns)
for col in num_features_for_stats:
    print(f"\n{col}:")
    print(f"Mean: {X[col].mean():.2f}")
    print(f"Variance: {X[col].var():.2f}")
    print(f"Standard Deviation: {X[col].std():.2f}")
    print(f"Min: {X[col].min():.2f}")
    print(f"25%: {X[col].quantile(0.25):.2f}")
    print(f"Median: {X[col].median():.2f}")
    print(f"75%: {X[col].quantile(0.75):.2f}")
    print(f"Max: {X[col].max():.2f}")

    ## If there are too many unique values, show only the top 10
    #if len(unique_values) > 10:
    #    print(f"Top 10 most common values:")
    #    for value, count in value_counts.head(10).items():
    #        percentage = value_percentages[value]
    #        print(f"{value}: {count} ({percentage:.2f}%)")
    #    print(f"... and {len(unique_values) - 10} more values")
    #else:
    #    for value, count in value_counts.items():
    #        percentage = value_percentages[value]
    #        print(f"{value}: {count} ({percentage:.2f}%)")


# Calculate correlation matrix for numerical features
print("\n" + "="*50)
print("CORRELATION MATRIX FOR NUMERICAL FEATURES")
print("="*50)

# Create a correlation matrix for numerical features (excluding ID columns)
X_numeric = X[num_features_for_stats]
corr_matrix = X_numeric.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(corr_matrix)

# Find the most correlated features
print("\nTop 10 Positive Correlations:")
# Get the upper triangle of the correlation matrix (to avoid duplicates)
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

# Sort by absolute correlation value (descending)
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# Print top 10 positive correlations
positive_corrs = [pair for pair in corr_pairs if pair[2] > 0]
for i, (col1, col2, corr) in enumerate(positive_corrs[:10]):
    print(f"{col1} - {col2}: {corr:.4f}")

# Perform chi-square test for the top 2 positive correlations
print("\nChi-Square Test for Top 2 Positive Correlations:")
try:
    from scipy import stats
    import numpy as np
    
    # Function to bin continuous data into categories for chi-square test
    def bin_data(data, bins=5):
        return pd.qcut(data, q=bins, labels=False, duplicates='drop')
    
    # Test for top 2 positive correlations
    for i, (col1, col2, corr) in enumerate(positive_corrs[:2]):
        print(f"\n{i+1}. {col1} - {col2} (Correlation: {corr:.4f})")
        
        # Bin the continuous data into categories
        try:
            # Try to create equal-sized bins
            binned_col1 = bin_data(X[col1])
            binned_col2 = bin_data(X[col2])
            
            # Create contingency table
            contingency = pd.crosstab(binned_col1, binned_col2)
            
            # Perform chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            print(f"Chi-square value: {chi2:.4f}")
            print(f"p-value: {p:.8f}")
            print(f"Degrees of freedom: {dof}")
            
            # Interpret the result
            alpha = 0.05
            print(f"Significance level: {alpha}")
            if p <= alpha:
                print(f"Result: Reject null hypothesis - There is a significant relationship between {col1} and {col2}")
            else:
                print(f"Result: Fail to reject null hypothesis - No significant relationship detected between {col1} and {col2}")
        except Exception as e:
            print(f"Error performing chi-square test: {e}")
    
    # Additional test for AGE_YEARS and AMT_CREDIT
    print("\nChi-Square Test for AGE_YEARS and AMT_CREDIT:")
    try:
        # Bin the continuous data into categories
        binned_age = bin_data(X['AGE_YEARS'])
        binned_credit = bin_data(X['AMT_CREDIT'])
        
        # Calculate correlation for reference
        age_credit_corr = X['AGE_YEARS'].corr(X['AMT_CREDIT'])
        print(f"Correlation between AGE_YEARS and AMT_CREDIT: {age_credit_corr:.4f}")
        
        # Create contingency table
        contingency = pd.crosstab(binned_age, binned_credit)
        
        # Perform chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        print(f"Chi-square value: {chi2:.4f}")
        print(f"p-value: {p:.8f}")
        print(f"Degrees of freedom: {dof}")
        
        # Interpret the result
        alpha = 0.05
        print(f"Significance level: {alpha}")
        if p <= alpha:
            print(f"Result: Reject null hypothesis - There is a significant relationship between AGE_YEARS and AMT_CREDIT")
        else:
            print(f"Result: Fail to reject null hypothesis - No significant relationship detected between AGE_YEARS and AMT_CREDIT")
    except Exception as e:
        print(f"Error performing chi-square test for AGE_YEARS and AMT_CREDIT: {e}")
    
    # Test for likely unrelated variables: HOUR_APPR_PROCESS_START and FLAG_EMAIL
    print("\nChi-Square Test for HOUR_APPR_PROCESS_START and FLAG_EMAIL (likely unrelated):")
    try:
        # Check if both variables exist in the dataset
        if 'HOUR_APPR_PROCESS_START' in X.columns and 'FLAG_EMAIL' in X.columns:
            # Calculate correlation for reference
            hour_email_corr = X['HOUR_APPR_PROCESS_START'].corr(X['FLAG_EMAIL'])
            print(f"Correlation between HOUR_APPR_PROCESS_START and FLAG_EMAIL: {hour_email_corr:.4f}")
            
            # Bin the continuous data into categories (HOUR_APPR_PROCESS_START)
            # FLAG_EMAIL is likely already binary (0/1)
            binned_hour = pd.cut(X['HOUR_APPR_PROCESS_START'], bins=5, labels=False)
            
            # Create contingency table
            contingency = pd.crosstab(binned_hour, X['FLAG_EMAIL'])
            
            # Perform chi-square test
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            print(f"Chi-square value: {chi2:.4f}")
            print(f"p-value: {p:.8f}")
            print(f"Degrees of freedom: {dof}")
            
            # Interpret the result
            alpha = 0.05
            print(f"Significance level: {alpha}")
            if p <= alpha:
                print(f"Result: Reject null hypothesis - There is a significant relationship between HOUR_APPR_PROCESS_START and FLAG_EMAIL")
            else:
                print(f"Result: Fail to reject null hypothesis - No significant relationship detected between HOUR_APPR_PROCESS_START and FLAG_EMAIL")
        else:
            print(f"One or both variables not found in dataset. Available columns: {', '.join(X.columns)}")
    except Exception as e:
        print(f"Error performing chi-square test for HOUR_APPR_PROCESS_START and FLAG_EMAIL: {e}")
    
  
except ImportError:
    print("SciPy not available. Skipping chi-square tests.")
except Exception as e:
    print(f"Error in chi-square testing: {e}")

# Print top 10 negative correlations
print("\nTop 10 Negative Correlations:")
negative_corrs = [pair for pair in corr_pairs if pair[2] < 0]
negative_corrs.sort(key=lambda x: x[2])
for i, (col1, col2, corr) in enumerate(negative_corrs[:10]):
    print(f"{col1} - {col2}: {corr:.4f}")

# Try to create a heatmap if matplotlib and seaborn are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nGenerating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    
    # Save the heatmap to a file
    heatmap_file = 'correlation_heatmap.png'
    plt.savefig(heatmap_file)
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_file}")
except ImportError:
    print("\nMatplotlib or Seaborn not available. Skipping heatmap generation.")
except Exception as e:
    print(f"\nError generating heatmap: {e}")

# Outlier Analysis for Numerical Features
print("\n" + "="*50)
print("OUTLIER ANALYSIS FOR NUMERICAL FEATURES")
print("="*50)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import pandas as pd
    
    # Suppress specific seaborn warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Function to detect outliers using IQR method
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        
        return {
            'column': column,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100,
            'min': df[column].min(),
            'max': df[column].max()
        }
    
    # Exclude ID columns and categorical-like numerical columns from outlier analysis
    exclude_columns = ['Unnamed: 0', 'SK_ID_CURR', 'REGION_RATING_CLIENT', 'IS_PROCESS_WEEKEND', 'OWN_CAR_AGE']
    outlier_features = [col for col in num_features if col not in exclude_columns]
    
    # Create a directory for outlier plots if it doesn't exist
    outlier_plots_dir = 'outlier_plots'
    if not os.path.exists(outlier_plots_dir):
        os.makedirs(outlier_plots_dir)
    
    # Create a list to store outlier analysis results
    outlier_results = []
    
    # Print header for outlier analysis results
    print("\nOutlier Analysis Results:")
    print("-" * 100)
    print(f"{'Column':<25} {'Outliers':<10} {'Percentage':<12} {'Min':<15} {'Max':<15} {'Lower Bound':<15} {'Upper Bound':<15}")
    print("-" * 100)
    
    # Analyze each numerical feature
    exclude_columns = ['Unnamed: 0', 'SK_ID_CURR', 'REGION_RATING_CLIENT', 'IS_PROCESS_WEEKEND', 'OWN_CAR_AGE']
    outlier_features = [col for col in num_features if col not in exclude_columns]
    
    for col in outlier_features:
        # Skip columns with all identical values
        if X[col].nunique() <= 1:
            continue
            
        # Detect outliers
        result = detect_outliers(X, col)
        
        # Add result to the list for later saving to table
        outlier_results.append({
            'Column': result['column'],
            'Outlier Count': result['outlier_count'],
            'Outlier Percentage': result['outlier_percentage'],
            'Min': result['min'],
            'Max': result['max'],
            'Lower Bound': result['lower_bound'],
            'Upper Bound': result['upper_bound'],
            'Q1': result['Q1'],
            'Q3': result['Q3'],
            'IQR': result['IQR']
        })
        
        # Print results
        print(f"{result['column']:<25} {result['outlier_count']:<10} {result['outlier_percentage']:.2f}% {result['min']:<15.2f} {result['max']:<15.2f} {result['lower_bound']:<15.2f} {result['upper_bound']:<15.2f}")
        
        # Create boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=X[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.savefig(f'{outlier_plots_dir}/{col}_boxplot.png')
        plt.close()
        
        # Create histogram with bounds marked
        plt.figure(figsize=(10, 6))
        sns.histplot(X[col], kde=True)
        plt.axvline(x=result['lower_bound'], color='r', linestyle='--', label=f'Lower bound: {result["lower_bound"]:.2f}')
        plt.axvline(x=result['upper_bound'], color='r', linestyle='--', label=f'Upper bound: {result["upper_bound"]:.2f}')
        plt.title(f'Distribution of {col} with Outlier Bounds')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{outlier_plots_dir}/{col}_distribution.png')
        plt.close()
    
    # Create a DataFrame from the outlier results
    outlier_df = pd.DataFrame(outlier_results)
    
    # Save the results to CSV and Excel files
    try:
        # Save to CSV
        outlier_df.to_csv('outlier_analysis_results.csv', index=False)
        print("\nOutlier analysis results saved to 'outlier_analysis_results.csv'")
        
        # Save to Excel with formatting
        try:
            # Try to save as Excel with formatting
            writer = pd.ExcelWriter('outlier_analysis_results.xlsx', engine='xlsxwriter')
            outlier_df.to_excel(writer, sheet_name='Outlier Analysis', index=False)
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Outlier Analysis']
            
            # Add some cell formats
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            number_format = workbook.add_format({'num_format': '#,##0.00'})
            
            # Write the column headers with the defined format
            for col_num, value in enumerate(outlier_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet.set_column('A:A', 20)  # Column name
            worksheet.set_column('B:K', 15)  # Other columns
            
            # Format the percentage column
            worksheet.set_column(2, 2, 15, percent_format)  # Outlier Percentage column
            
            # Close the writer
            writer.close()
            print("Outlier analysis results saved to 'outlier_analysis_results.xlsx'")
        except Exception as excel_error:
            print(f"Could not save Excel file with formatting: {excel_error}")
            # Try simple Excel save without formatting
            outlier_df.to_excel('outlier_analysis_results.xlsx', index=False)
            print("Outlier analysis results saved to 'outlier_analysis_results.xlsx' (without formatting)")
    except Exception as save_error:
        print(f"Error saving outlier analysis results: {save_error}")
    
    print("\nOutlier analysis completed. Boxplots and distribution plots saved in the 'outlier_plots' directory.")
    
except ImportError:
    print("Matplotlib or seaborn not available. Skipping outlier analysis visualization.")
except Exception as e:
    print(f"Error during outlier analysis: {e}")

# Remove outliers that are more than 3 IQR away from Q1 or Q3
print("\nRemoving extreme outliers (beyond 3 IQR)...")
try:
    import numpy as np
    
    # Variables to clean outliers from
    target_vars = ['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'INCOME_PER_PERSON_IN_FAMILY']
    
    # Store original statistics and dataset size for comparison
    original_stats = {}
    original_size = len(X)
    
    for col in target_vars:
        original_stats[col] = {
            'mean': X[col].mean(),
            'std': X[col].std(),
            'min': X[col].min(),
            'max': X[col].max(),
            'count': len(X)
        }
    
    # Function to identify extreme outliers (beyond 3 IQR)
    def identify_extreme_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        return (df[column] >= lower_bound) & (df[column] <= upper_bound)
    
    # Apply outlier removal for each variable
    for col in target_vars:
        # Get mask of non-outlier values
        mask = identify_extreme_outliers(X, col)
        
        # Count outliers
        outliers_count = (~mask).sum()
        outliers_percentage = (outliers_count / len(X)) * 100
        
        # Remove outliers
        X = X[mask]
        
        # Print results
        print(f"\n{col}:")
        print(f"Outliers removed: {outliers_count} ({outliers_percentage:.2f}%)")
        print(f"Original mean: {original_stats[col]['mean']:.2f}, New mean: {X[col].mean():.2f}")
        print(f"Original std: {original_stats[col]['std']:.2f}, New std: {X[col].std():.2f}")
        print(f"Original min: {original_stats[col]['min']:.2f}, New min: {X[col].min():.2f}")
        print(f"Original max: {original_stats[col]['max']:.2f}, New max: {X[col].max():.2f}")
    
    # Print overall reduction in dataset size
    rows_removed = original_size - len(X)
    percentage_removed = (rows_removed / original_size) * 100
    print(f"\nTotal rows removed: {rows_removed} ({percentage_removed:.2f}%)")
    print(f"Original dataset size: {original_size}, New dataset size: {len(X)}")

except Exception as e:
    print(f"Error during outlier removal: {e}")

# Feature Importance Analysis using Random Forest
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\nPerforming feature importance analysis using Random Forest...")
    
    # Check if TARGET column exists in the original dataframe
    if 'TARGET' in df.columns:
        # Extract target variable
        y = df['TARGET']
        
        # Make sure X and y have the same number of samples after outlier removal
        common_index = X.index.intersection(df.loc[df['TARGET'].notna()].index)
        X_for_model = X.loc[common_index].copy()
        y_for_model = df.loc[common_index, 'TARGET'].copy()
        
        print(f"\nUsing {len(X_for_model)} samples for feature importance analysis")
        
        # Select features for analysis (exclude ID columns)
        exclude_cols = ['Unnamed: 0', 'SK_ID_CURR', 'ID']
        features = [col for col in X_for_model.columns if col not in exclude_cols]
        X_for_model = X_for_model[features]
        
        # Handle categorical features
        cat_features = X_for_model.select_dtypes(include=['object']).columns.tolist()
        
        # Apply label encoding to categorical features
        for col in cat_features:
            le = LabelEncoder()
            X_for_model[col] = le.fit_transform(X_for_model[col].astype(str))
        
        # Fill missing values with median for numerical and mode for categorical
        for col in X_for_model.columns:
            if X_for_model[col].dtype in ['int64', 'float64']:
                X_for_model[col] = X_for_model[col].fillna(X_for_model[col].median())
            else:
                X_for_model[col] = X_for_model[col].fillna(X_for_model[col].mode()[0])
        
        # Train a Random Forest model
        print("Training Random Forest model to determine feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_for_model, y_for_model)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print feature ranking
        print("\nFeature ranking:")
        for f in range(min(20, X_for_model.shape[1])):
            print(f"{f+1}. {X_for_model.columns[indices[f]]} ({importances[indices[f]]:.4f})")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(min(20, X_for_model.shape[1])), importances[indices[:20]], align="center")
        plt.xticks(range(min(20, X_for_model.shape[1])), [X_for_model.columns[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("\nFeature importance analysis completed. Plot saved to 'feature_importance.png'.")
    else:
        print("\nTARGET column not found in the dataset. Skipping feature importance analysis.")
        print("To perform this analysis, ensure your dataset includes the target variable.")

except ImportError:
    print("Required libraries (scikit-learn) not available. Skipping feature importance analysis.")
except Exception as e:
    print(f"Error during feature importance analysis: {e}")


# PCA Visualization for 2D representation of the dataset
print("\nGenerating 2D PCA visualization of the dataset...")
try:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.preprocessing import StandardScaler
    
    # Set style for professional visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Check if we have the processed data with target variable
    if target and 'TARGET' in df.columns:
        # Prepare data for PCA (use only numerical features)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        X_numeric = df[numeric_cols].drop('TARGET', axis=1, errors='ignore')
        
        # Handle any remaining NaN values
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Filter outliers for better visualization
        z_scores = np.abs((X_numeric - X_numeric.mean()) / X_numeric.std())
        mask = (z_scores < 3).all(axis=1)
        
        X_filtered = X_numeric[mask]
        y_filtered = df['TARGET'][mask]
        
        print(f"Filtered {len(X_numeric) - len(X_filtered)} outliers for PCA visualization")
        print(f"Using {len(X_filtered)} samples for visualization")
        
        # Standardize the data for better PCA results
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate density for better visualization
        from scipy.stats import gaussian_kde
        
        # Create figure with a specific size and high-quality settings
        plt.figure(figsize=(12, 10), dpi=100)
        
        # Create custom colormaps for better visualization
        default_cmap = LinearSegmentedColormap.from_list('default_cmap', ['#ffcccb', '#ff0000'])
        non_default_cmap = LinearSegmentedColormap.from_list('non_default_cmap', ['#cce5ff', '#0066cc'])
        
        # Plot points by target class with enhanced aesthetics
        default_mask = y_filtered == 1
        non_default_mask = y_filtered == 0
        
        # Plot non-default points (majority class)
        plt.scatter(X_pca[non_default_mask, 0], X_pca[non_default_mask, 1], 
                   color='#0066cc', alpha=0.5, label='No Default', s=25, 
                   edgecolor='white', linewidth=0.2)
        
        # Plot default points (minority class) with higher visibility
        plt.scatter(X_pca[default_mask, 0], X_pca[default_mask, 1], 
                   color='#ff0000', alpha=0.7, label='Default', s=35, 
                   edgecolor='white', linewidth=0.2)
        
        # Add contour lines to show density (optional)
        try:
            # For non-default class
            non_default_data = X_pca[non_default_mask]
            if len(non_default_data) > 50:  # Only if we have enough points
                x = non_default_data[:, 0]
                y = non_default_data[:, 1]
                k = gaussian_kde(np.vstack([x, y]))
                xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                plt.contour(xi, yi, zi, levels=5, colors=['#0066cc'], alpha=0.3, linewidths=0.5)
            
            # For default class
            default_data = X_pca[default_mask]
            if len(default_data) > 50:  # Only if we have enough points
                x = default_data[:, 0]
                y = default_data[:, 1]
                k = gaussian_kde(np.vstack([x, y]))
                xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
                plt.contour(xi, yi, zi, levels=5, colors=['#ff0000'], alpha=0.3, linewidths=0.5)
        except Exception as e:
            print(f"Could not generate density contours: {e}")
        
        # Add title and labels with enhanced styling
        plt.title('Principal Component Analysis: Home Credit Default Risk', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})', 
                 fontsize=12, fontweight='bold')
        plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})', 
                 fontsize=12, fontweight='bold')
        
        # Add a text box with total explained variance
        total_var = sum(pca.explained_variance_ratio_)
        plt.annotate(f'Total Explained Variance: {total_var:.3f}', 
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=11, ha='left', va='top')
        
        # Add class distribution information
        default_count = np.sum(default_mask)
        non_default_count = np.sum(non_default_mask)
        total_count = len(y_filtered)
        
        plt.annotate(f'Class Distribution:\nDefault: {default_count} ({default_count/total_count:.1%})\n'
                    f'No Default: {non_default_count} ({non_default_count/total_count:.1%})', 
                    xy=(0.02, 0.90), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=11, ha='left', va='top')
        
        # Enhanced legend with custom styling
        legend = plt.legend(loc='upper right', frameon=True, framealpha=0.9, 
                          fontsize=12, title="Credit Status", title_fontsize=13)
        legend.get_frame().set_edgecolor('gray')
        
        # Add subtle grid lines
        plt.grid(True, linestyle='--', alpha=0.2, color='gray')
        
        # Add a border around the plot
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        for spine in plt.gca().spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)
        
        # Adjust layout and save with high quality
        plt.tight_layout()
        plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
        print("Enhanced PCA visualization saved as 'pca_visualization.png'")
        
        # Print explained variance information
        print(f"Explained variance ratio by the first two principal components: {total_var:.3f}")
        print("PCA analysis complete.")
        
    else:
        print("TARGET column not found. Cannot create PCA visualization with target coloring.")
        
except ImportError:
    print("Required libraries (sklearn.decomposition.PCA, matplotlib, or scipy) not available. Skipping PCA visualization.")
except Exception as e:
    print(f"Error during PCA visualization: {e}")

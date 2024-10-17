import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats

# File names
DATA_FILE_1 = 'dataset1.csv'
DATA_FILE_2 = 'dataset2.csv'
DATA_FILE_3 = 'dataset3.csv'
DATA_FILE_MERGED = 'merged_data.csv'
OPTIMIZED_DATA_FILE = 'optimizedData.csv'

# Load data files
if not os.path.exists(DATA_FILE_MERGED):
    dataset1 = pd.read_csv(DATA_FILE_1)
    dataset2 = pd.read_csv(DATA_FILE_2)
    dataset3 = pd.read_csv(DATA_FILE_3)

    # Merging the datasets on 'ID'
    merged_data = pd.merge(dataset1, dataset2, on='ID', how='inner')
    merged_data = pd.merge(merged_data, dataset3, on='ID', how='inner')
    merged_data.to_csv(DATA_FILE_MERGED, index=False)
else:
    # Load the merged dataset
    merged_data = pd.read_csv('merged_data.csv')

# Calculate the well-being score as the mean of specific columns
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_data['well_being_score'] = merged_data[wellbeing_columns].mean(axis=1)

# Independent variables (X) and Dependent variable (Y)
X_columns = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
Y_column = 'well_being_score'

# Remove outliers using the IQR method for each column
for column in X_columns + [Y_column]:
    Q1 = merged_data[column].quantile(0.25)
    Q3 = merged_data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data for the current column to remove outliers
    merged_data = merged_data[(merged_data[column] >= lower_bound) & (merged_data[column] <= upper_bound)]

# Update X and Y without outliers (IQR method)
X_no_outliers_iqr = merged_data[X_columns]
Y_no_outliers_iqr = merged_data[Y_column]

# Add constant for the linear regression
X_no_outliers_iqr = sm.add_constant(X_no_outliers_iqr)

# Build the linear regression model without outliers (IQR method)
model_no_outliers_iqr = sm.OLS(Y_no_outliers_iqr, X_no_outliers_iqr).fit()
print("\nModel Summary without Outliers (IQR method):")
print(model_no_outliers_iqr.summary())

######### Apply power transformation after removing outliers #############
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer()
X = X_no_outliers_iqr.drop(['const'], axis=1)
X_power = scaler.fit_transform(X.values)
df_X_power = pd.DataFrame(X_power, index=X.index, columns=X.columns)

#save optimized x and y in a new file 
optimized_data = pd.concat([df_X_power, Y_no_outliers_iqr], axis=1)
optimized_data.to_csv(OPTIMIZED_DATA_FILE, index=False)

#Re-build the linear regression using statemodels using statsmodels + standardisation
df_X_power = sm.add_constant(df_X_power)
model = sm.OLS(Y_no_outliers_iqr, df_X_power).fit()
model_report = model.summary()
print(model_report)
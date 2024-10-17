import os
import pandas as pd
import statsmodels.api as sm

#file names
DATA_FILE_1 = 'dataset1.csv'
DATA_FILE_2 = 'dataset2.csv'
DATA_FILE_3 = 'dataset3.csv'
DATA_FILE_MERGED = 'merged_data.csv'

#load data files
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

wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_data['well_being_score'] = merged_data[wellbeing_columns].mean(axis=1)

X = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]  # Independent variables
Y = merged_data['well_being_score']  # Dependent variable (well-being score)

#build the linear regression using statemodels
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_report = model.summary()
print(model_report)

#apply Power Transformer
from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer()
X = X.drop(['const'], axis=1)
X_power = scaler.fit_transform(X.values)
df_X_power = pd.DataFrame(X_power, index=X.index, columns=X.columns)

#Re-build the linear regression using statemodels using statsmodels + standardisation
df_X_power = sm.add_constant(df_X_power)
model = sm.OLS(Y, df_X_power).fit()
model_report = model.summary()
print(model_report)

###### A slight improvement can be seen in the model R-squared from 0.042 to 0.043 ######
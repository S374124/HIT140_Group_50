import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns


########### Check for non-linear explanatory variables ###########

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

print(merged_data.head())

X = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]  # Independent variables
Y = merged_data['well_being_score']  # Dependent variable (well-being score)

#build the linear regression using statemodels
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_report = model.summary()
print(model_report)


#plot variables to check linearity
# Plot each independent variable against the well-being score
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Scatter Plots of Independent Variables vs. Well-being Score")
axs = axs.ravel()

for i, col in enumerate(X.columns[1:]):  # Skipping the constant column
    axs[i].scatter(merged_data[col], Y)
    axs[i].set_xlabel(col)
    axs[i].set_ylabel("Well-being Score")
    axs[i].set_title(f"{col} vs Well-being Score")

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the title
plt.show()

#### All looks good. There are no non-linear explanatory variables ###
#### Therefore, no need to apply log transform #####


#apply log-transform

# merged_data = merged_data[merged_data['T_wk'] > 0] #remove minus or NaN values
# merged_data["LOG_T_WK"] = merged_data["T_wk"].apply(np.log)
# merged_data = merged_data.drop('T_wk', axis=1)
# x = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'LOG_T_WK']]
# y = merged_data['well_being_score']

# print(merged_data.head())

# #build the linear regression using statemodels after applying log
# x = sm.add_constant(x)
# model = sm.OLS(y, x).fit()
# model_report = model.summary()
# print(model_report)
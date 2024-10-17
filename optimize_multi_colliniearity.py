import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt



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
merged_data['wb_score'] = merged_data[wellbeing_columns].mean(axis=1)
df = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk', 'wb_score']]

#compute the correlation matrix
corr = df.corr()

#plot the matrix as a heatmap
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=False, annot=True)

#customize the labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()

#build the original multiple linear model
X = df.iloc[:, :-1]  # Independent variables
Y = df.iloc[:,-1]  # Dependent variable (well-being score)

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model_report = model.summary()
print(model_report)


########### To Remove multi Collinearity by removing variables ##############

#remove "C_wk" variable and build the model again ## uncomment only one at a time ##
# df = df.drop(['C_wk'], axis=1)
# X = df.iloc[:, :-1]  # Independent variables
# Y = df.iloc[:,-1]  # Dependent variable (well-being score)

# X = sm.add_constant(X)
# model = sm.OLS(Y, X).fit()
# model_report = model.summary()
# print(model_report)

#remove "C_we" variable and build the model again  ## uncomment only one at a time ##
# df = df.drop(['C_we'], axis=1)
# X = df.iloc[:, :-1]  # Independent variables
# Y = df.iloc[:,-1]  # Dependent variable (well-being score)

# X = sm.add_constant(X)
# model = sm.OLS(Y, X).fit()
# model_report = model.summary()
# print(model_report)


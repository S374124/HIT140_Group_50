import os
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
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

#print(merged_data.head())

wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_data['well_being_score'] = merged_data[wellbeing_columns].mean(axis=1)

X = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].values  # Independent variables
y = merged_data['well_being_score'].values  # Dependent variable (well-being score)

# Split dataset into 60% training and 40% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: %.2f" % model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set based on X_test
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)
r_2 = metrics.r2_score(y_test, y_pred)

# Display performance metrics
print("Linear Regression performance:")
print("MAE: %.2f" % mae)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % rmse)
print("RMSE (Normalised): %.2f" % rmse_norm)
print("R^2: %.3f" % r_2)


#Baseline Model
y_base = np.mean(y_train)
y_pred_base = [y_base] * len(y_test)

# Compute standard performance metrics of the baseline model:
mae = metrics.mean_absolute_error(y_test, y_pred_base)
mse = metrics.mean_squared_error(y_test, y_pred_base)
rmse = math.sqrt(mse)
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)
r_2 = metrics.r2_score(y_test, y_pred_base)

# Display performance metrics
print("Baseline model performance:")
print("MAE: %.2f" % mae)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % rmse)
print("RMSE (Normalised): %.2f" % rmse_norm)
print("R^2: %.2f" % r_2)

######### Display in a plot #########

df_sample = df_pred.head(50)

# Draw a line chart to display the actual and predicted values for the sampled data
plt.figure(figsize=(10,6))

# Plotting actual values
plt.plot(df_sample['Actual'].values, label='Actual', color='blue', marker='o')

# Plotting predicted values
plt.plot(df_sample['Predicted'].values, label='Predicted', color='green', marker='x')

# Adding labels and title
plt.title('Actual vs Predicted Well-being Scores (Sample of 50)')
plt.xlabel('Data Points (First 50 Records)')
plt.ylabel('Well-being Score')
plt.legend()

# Display the chart
plt.show()
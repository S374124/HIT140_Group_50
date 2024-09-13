import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Merge the datasets on 'ID'
merged_data = pd.merge(pd.merge(dataset1, dataset2, on='ID'), dataset3, on='ID')

# Display the first few rows to ensure the data is loaded correctly
print(merged_data.head())

# List of well-being indicators
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Calculate the composite well-being score
merged_data['Wellbeing_Score'] = merged_data[wellbeing_columns].mean(axis=1)

# Display the first few rows to ensure the score is calculated correctly
print(merged_data[['Wellbeing_Score']].head())

# Calculate the mean of the well-being score
mean_wellbeing = merged_data['Wellbeing_Score'].mean()

# Calculate the standard error of the mean
std_error = stats.sem(merged_data['Wellbeing_Score'])

print(f"Mean Well-being Score: {mean_wellbeing}")
print(f"Standard Error: {std_error}")

# Calculate the 95% confidence interval
confidence_interval = stats.t.interval(0.95, len(merged_data['Wellbeing_Score'])-1, loc=mean_wellbeing, scale=std_error)

print(f"95% Confidence Interval for Well-being Score: {confidence_interval}")

plt.figure(figsize=(10, 6))

# Density plot
sns.kdeplot(merged_data['Wellbeing_Score'], fill=True, color="skyblue", alpha=0.4, linewidth=3)

# Mean and Confidence Interval
plt.axvline(mean_wellbeing, color='gray', linestyle='--', label=f'Mean: {mean_wellbeing:.2f}')
plt.fill_betweenx([0, 1], confidence_interval[0], confidence_interval[1], color='orange', alpha=0.3, label='95% Confidence Interval')

plt.title('Density Plot of Well-being Scores with 95% Confidence Interval')
plt.xlabel('Well-being Score')
plt.ylabel('Density')
plt.legend()

plt.show()





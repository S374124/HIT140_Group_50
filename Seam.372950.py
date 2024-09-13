import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# Load datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Merge datasets
merged_data = pd.merge(dataset1, dataset2, on='ID')
complete_data = pd.merge(merged_data, dataset3, on='ID')

# Calculate total screen time per day (weekday and weekend combined)
complete_data['Total_Screen_Time'] = complete_data[['C_wk', 'G_wk', 'S_wk', 'T_wk', 'C_we', 'G_we', 'S_we', 'T_we']].sum(axis=1)

# Calculate mean well-being score across all indicators
complete_data['Mean_Well_Being'] = complete_data.iloc[:, -14:].mean(axis=1)

# Plot histogram of well-being scores
plt.hist(complete_data['Mean_Well_Being'], bins=30, edgecolor='black')
plt.title('Histogram of Well-Being Scores')
plt.xlabel('Well-Being Score')
plt.ylabel('Frequency')
plt.show()

# Split into high and low screen time groups based on the median
median_screen_time = complete_data['Total_Screen_Time'].median()
high_screen_time_group = complete_data[complete_data['Total_Screen_Time'] > median_screen_time]
low_screen_time_group = complete_data[complete_data['Total_Screen_Time'] <= median_screen_time]

# Calculate means and standard deviations
mean_high = high_screen_time_group['Mean_Well_Being'].mean()
std_high = high_screen_time_group['Mean_Well_Being'].std()

mean_low = low_screen_time_group['Mean_Well_Being'].mean()
std_low = low_screen_time_group['Mean_Well_Being'].std()

# Perform independent t-test
t_stat, p_val = st.ttest_ind(high_screen_time_group['Mean_Well_Being'], low_screen_time_group['Mean_Well_Being'])

# Print results
print(f'Mean Well-Being (High Screen Time): {mean_high:.2f}, Standard Deviation: {std_high:.2f}')
print(f'Mean Well-Being (Low Screen Time): {mean_low:.2f}, Standard Deviation: {std_low:.2f}')
print(f't-statistic: {t_stat:.2f}, p-value: {p_val:.2f}')

# Conclusion based on p-value
if p_val < 0.05:
    print("Reject the null hypothesis: There is a significant difference in well-being scores between high and low screen time groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in well-being scores between high and low screen time groups.")

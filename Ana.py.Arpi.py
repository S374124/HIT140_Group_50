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

# Focus on specific screen time: Smartphone usage only
complete_data['Total_Smartphone_Time'] = complete_data['S_wk'] + complete_data['S_we']

# Calculate the mean score for feeling cheerful as a well-being indicator
complete_data['Cheer_Score'] = complete_data['Cheer']

# Plot histogram of cheerful scores
plt.hist(complete_data['Cheer_Score'], bins=20, edgecolor='black')
plt.title('Histogram of Cheerful Scores')
plt.xlabel('Cheerful Score')
plt.ylabel('Frequency')
plt.show()

# Split into high and low smartphone time groups based on the median
median_smartphone_time = complete_data['Total_Smartphone_Time'].median()
high_smartphone_time_group = complete_data[complete_data['Total_Smartphone_Time'] > median_smartphone_time]
low_smartphone_time_group = complete_data[complete_data['Total_Smartphone_Time'] <= median_smartphone_time]

# Calculate means and standard deviations
mean_high_cheer = high_smartphone_time_group['Cheer_Score'].mean()
std_high_cheer = high_smartphone_time_group['Cheer_Score'].std()

mean_low_cheer = low_smartphone_time_group['Cheer_Score'].mean()
std_low_cheer = low_smartphone_time_group['Cheer_Score'].std()

# Perform independent t-test
t_stat_cheer, p_val_cheer = st.ttest_ind(high_smartphone_time_group['Cheer_Score'], low_smartphone_time_group['Cheer_Score'])

# Print results
print(f'Mean Cheerful Score (High Smartphone Time): {mean_high_cheer:.2f}, Standard Deviation: {std_high_cheer:.2f}')
print(f'Mean Cheerful Score (Low Smartphone Time): {mean_low_cheer:.2f}, Standard Deviation: {std_low_cheer:.2f}')
print(f't-statistic: {t_stat_cheer:.2f}, p-value: {p_val_cheer:.2f}')

# Conclusion based on p-value
if p_val_cheer < 0.05:
    print("Reject the null hypothesis: There is a significant difference in cheerful scores between high and low smartphone time groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in cheerful scores between high and low smartphone time groups.")

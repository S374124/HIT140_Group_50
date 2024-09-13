import os
import pandas as pd
import matplotlib.pyplot as plot
import scipy.stats as st

#Null Hypothesis: The well-being score of the high screen time group is greater than or equal to the low screen time group.
#Alternative Hypothesis: The well-being score of the high screen time group is less than the low screen time group.

#file names
DATA_FILE_1 = 'dataset1.csv'
DATA_FILE_2 = 'dataset2.csv'
DATA_FILE_3 = 'dataset3.csv'
DATA_FILE_MERGED = 'merged_data.csv'
ACCEPTABLE_P_VAL = 0.05
HO =  'The well-being score of the high screen time group is greater than or equal to the low screen time group.'
H1 = 'The well-being score of the high screen time group is less than the low screen time group.'

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
    merged_data = pd.read_csv(DATA_FILE_MERGED)

#get mean of well being data in each row, to combine all required columns
well_being_scores = merged_data.iloc[:, 12:].mean(axis=1)  #starting from col index 12 (13th col) to last col, mean because to get an average behaviour across columns

#plot to verify whether the data has a normal destribution or not
plot.hist(well_being_scores, bins=30, edgecolor='black')
plot.title('Histogram of Well-being Scores')
plot.xlabel('Well-being Score')
plot.ylabel('Frequency')
plot.show()

#Devide into two groups

total_screen_time = merged_data.iloc[:, 4:12].sum(axis=1) #get sum of each column starting from col index 4 (5th col) to col index 11 (12th col) 
median_screen_time = total_screen_time.median() #to define a break point
high_screen_time_group = well_being_scores[total_screen_time > median_screen_time] #two arrays need to be same length, then it will match according to the indexes of both arrays
low_screen_time_group = well_being_scores[total_screen_time <= median_screen_time]

# Calculate basic statistics

#high screen time group
x_bar_well_being_high = st.tmean(high_screen_time_group) #sample mean
std_well_being_high = st.tstd(high_screen_time_group) #standard deviation
length_high = len(high_screen_time_group)

print('Mean (Group High): %.2f' % x_bar_well_being_high)
print('Standard Deviation  (Group High): %.2f' % std_well_being_high)
print('Sample size (Group High): ', length_high)

#low screen time group
x_bar_well_being_low = st.tmean(low_screen_time_group) #sample mean
std_well_being_low = st.tstd(low_screen_time_group) #standard deviation
length_low = len(low_screen_time_group)

print('Mean (Group Low): %.2f' % x_bar_well_being_low)
print('Standard Deviation  (Group Low): %.2f' % std_well_being_low)
print('Sample size (Group Low): ', length_low)

#t-test

t_stats, p_val = st.ttest_ind_from_stats(x_bar_well_being_high, std_well_being_high, length_high, 
                                            x_bar_well_being_low, std_well_being_low, length_low,
                                            equal_var=False, alternative='less') #two sample check

print('t_stats: %.2f' %t_stats)
print('p_val: %.2f' %p_val)

if p_val < ACCEPTABLE_P_VAL:
    print(f"There is no enough evidence to accept the null hypothesis (p-value = {p_val:.5f}).")
    print('Therefore, ', H1)
else:
    print(f"There is enough evidence to accept the null hypothesis (p-value = {p_val:.5f}).")
    print('Therefore, ', HO)

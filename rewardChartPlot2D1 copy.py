import pandas as pd

# Load the CSV file
df = pd.read_csv('finalresults_with_rewards.csv')

# Sort the DataFrame by the 'AverageReward' column in descending order
sorted_df = df.sort_values(by='AverageReward', ascending=False)

# Save the sorted DataFrame back to a CSV file
sorted_df.to_csv('sorted_finalresults_with_rewards.csv', index=False)

print("CSV sorted and saved as 'sorted_finalresults_with_rewards.csv'")

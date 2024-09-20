import os
import pandas as pd

import pandas as pd



def calculate_average_rewards(jobId):
    base_path = "./results"  # Replace this with the path to your folders

    folder_rewards = []
    
    # Iterate over all folders in the base path
    for folder_name in os.listdir(base_path):
        if folder_name.startswith("output_"+jobId):  # Check if the folder name starts with 'output_'
            total_reward = 0
            run_count = 0
            
            folder_path = os.path.join(base_path, folder_name)
            
            # Iterate over all files in the folder
            for file_name in os.listdir(folder_path):
                if file_name.startswith("Wildfire_AOR_POMCPPF-") and file_name.endswith(".csv"):
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Sum all rewards for this run
                    total_reward += df['Reward'].sum()
                    run_count += 1
            
            # Calculate the average reward for the folder
            if run_count > 0:
                average_reward = total_reward / run_count
                return average_reward
                folder_rewards.append((folder_name, average_reward))
    
    # return folder_rewards

# Usage
# base_directory = "./results"  # Replace this with the path to your folders
# average_rewards = calculate_average_rewards(base_directory)

# # Print the results
# for folder, avg_reward in average_rewards:
#     print(f"Folder: {folder}, Average Reward: {avg_reward}")



# Read data from CSV files
job_info_df = pd.read_csv('https://raw.githubusercontent.com/bboyfury/pomcppf_experiment_1/main/job_info_output.csv')
sacct_df = pd.read_csv('https://raw.githubusercontent.com/bboyfury/pomcppf_experiment_1/main/job_statistics_output.csv')

# Ensure JobID columns are strings
job_info_df['JobID'] = job_info_df['JobID'].astype(str)
sacct_df['JobID'] = sacct_df['JobID'].astype(str)

# Extract the main JobID (before any dots or underscores)
sacct_df['MainJobID'] = sacct_df['JobID'].str.split(r'[._]').str[0]
job_info_df['MainJobID'] = job_info_df['JobID'].str.split(r'[._]').str[0]

# Merge the two dataframes on 'MainJobID'
merged_df = pd.merge(job_info_df, sacct_df, on='MainJobID')


# Convert necessary columns to numeric types
numeric_columns = ['TRAJECTORIES', 'Horizon', 'PARTICLES', 'ElapsedRaw']
for col in numeric_columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Apply the function to convert MaxRSS to bytes

# Convert MaxRSS_Bytes to MB (Megabytes)
merged_df['AverageReward'] = 0.0
merged_df['AverageReward'] = merged_df['MainJobID'].apply(calculate_average_rewards)
# Remove rows where 'MainJobID' does not contain 'batch'
# Loop through each unique JobID
merged_df = merged_df[merged_df['MaxRSS'].notna() & (merged_df['MaxRSS'] != '')]
# for i,result in merged_df.iterrows():
#     merged_df.at[i,'AverageReward'] =calculate_average_rewards(result['MainJobID'])

#     value=[reward for jobId, reward in average_rewards if jobId.strip("output_") == result['MainJobID']]
#     merged_df['AverageReward'] =value[0]
    
final_df = merged_df[['MainJobID', 'ElapsedRaw', 'MaxRSS', 'TRAJECTORIES', 'Horizon', 'PARTICLES','AverageReward']]
merged_df
final_df['MaxRSS'] = final_df['MaxRSS']
final_df['ElapsedRaw'] = final_df['ElapsedRaw'].round(2)
final_df['AverageReward'] = final_df['AverageReward'].round(2)
final_df.to_csv('finalresults_with_rewards.csv', index=False)
x=2
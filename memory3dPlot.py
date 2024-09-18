from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers import parse_maxrss



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

merged_df['MaxRSS_Bytes'] = merged_df['MaxRSS'].apply(parse_maxrss)

# Drop rows with missing values in key columns
merged_df.dropna(subset=['TRAJECTORIES', 'Horizon', 'PARTICLES', 'ElapsedRaw', 'MaxRSS_Bytes', 'FolderSize'], inplace=True)

grouped_particles_with_values = merged_df.groupby('PARTICLES').apply(
    lambda df: {
        'x': sorted(set(df['Horizon'].tolist())),
        'y': sorted(set(df['TRAJECTORIES'].tolist())),
        'value': [df.loc[(df['Horizon'] == h) & (df['TRAJECTORIES'] == t), 'MaxRSS_Bytes'].max() 
                  for h, t in zip(sorted(set(df['Horizon'])), sorted(set(df['TRAJECTORIES'])))]
    }
).to_dict()



# Convert necessary columns to numeric types
numeric_columns = ['TRAJECTORIES', 'Horizon', 'PARTICLES', 'ElapsedRaw']
for col in numeric_columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Apply the function to convert MaxRSS to bytes
merged_df['MaxRSS_Bytes'] = merged_df['MaxRSS'].apply(parse_maxrss)

# Convert MaxRSS_Bytes to MB (Megabytes)
merged_df['MaxRSS_MB'] = merged_df['MaxRSS_Bytes'] / (1024 * 1024)

# Drop rows with missing values in key columns
merged_df.dropna(subset=['TRAJECTORIES', 'Horizon', 'MaxRSS_MB'], inplace=True)

# Normalize MaxRSS_MB for color mapping
norm = Normalize(vmin=merged_df['MaxRSS_MB'].min(), vmax=merged_df['MaxRSS_MB'].max())

# Now, we create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapping (red for high, blue for mid, green for low)
sc = ax.scatter(merged_df['Horizon'], merged_df['TRAJECTORIES'], merged_df['MaxRSS_MB'], 
                c=merged_df['MaxRSS_MB'], cmap='RdYlGn_r', norm=norm, marker='o')

# Add a color bar
plt.colorbar(sc, ax=ax, label='MaxRSS (MB)')

# Set labels
ax.set_xlabel('Horizon')
ax.set_ylabel('TRAJECTORIES')
ax.set_zlabel('MaxRSS (MB)')

# Customize the view angle
ax.view_init(elev=30., azim=30)

# Show the plot
plt.show()
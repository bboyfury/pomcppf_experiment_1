import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

# Helper functions
def parse_maxrss(value):
    """Parses MaxRSS values like '1234K', '567M' into bytes."""
    if pd.isnull(value):
        return np.nan
    try:
        value = str(value)
        if value.endswith('K'):
            return float(value[:-1]) * 1024
        elif value.endswith('M'):
            return float(value[:-1]) * 1024 * 1024
        elif value.endswith('G'):
            return float(value[:-1]) * 1024 * 1024 * 1024
        else:
            return float(value)
    except:
        return np.nan

def bytes_to_human_readable(num, suffix='B'):
    """Converts bytes to human-readable format."""
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)

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
# Convert MaxRSS from bytes to MB (1 MB = 1024 * 1024 bytes)
for particle, data in grouped_particles_with_values.items():
    data['value_mb'] = [v / (1024 * 1024) for v in data['value']]

# Plotting the data with MaxRSS in MB
plt.figure(figsize=(10, 6))

for particle, data in grouped_particles_with_values.items():
    plt.plot(data['x'], data['value_mb'], marker='o', label=f'Particles: {particle}')

plt.xlabel('Horizon')
plt.ylabel('MaxRSS (MB)')
plt.title('MaxRSS vs Horizon for different Particle settings (in MB)')
plt.legend(title='Particles')
plt.grid(True)

plt.show()
plt.savefig('particle&Horizon.png')

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from helpers import bytes_to_human_readable, parse_maxrss



# Read data from CSV files
job_info_df = pd.read_csv('https://raw.githubusercontent.com/bboyfury/pomcppf_experiment_1/main/job_info_output.csv')
sacct_df = pd.read_csv('https://raw.githubusercontent.com/bboyfury/pomcppf_experiment_1/main/job_statistics_output.csv')  # Replace with your actual filename

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

# Assuming 'FolderSize' is available in the merged_df


# Drop rows with missing values in key columns
merged_df.dropna(subset=['TRAJECTORIES', 'Horizon', 'PARTICLES', 'ElapsedRaw', 'MaxRSS_Bytes', 'FolderSize'], inplace=True)

# Prepare 3D scatter data
x_data = merged_df['TRAJECTORIES'].values
y_data = merged_df['Horizon'].values
z_data = merged_df['PARTICLES'].values
# Prepare custom data to include the 'MaxRSS' column for hover info
# custom_data = merged_df[['MaxRSS']].values
custom_data = merged_df[['MaxRSS', 'JobID_x']].values
# Apply the helper function to convert MaxRSS_Bytes to a human-readable format

# Apply the helper function to convert MaxRSS_Bytes to a human-readable format
merged_df['MaxRSS_HumanReadable'] = merged_df['MaxRSS_Bytes'].apply(bytes_to_human_readable)

# Prepare 3D scatter data for MaxRSS plot
x_data = merged_df['TRAJECTORIES'].values
y_data = merged_df['Horizon'].values
z_data = merged_df['PARTICLES'].values
maxrss_bytes = merged_df['MaxRSS_Bytes'].values
id = merged_df['JobID_x'].values
maxrss_human_readable = merged_df['MaxRSS_HumanReadable'].values
maxrss_mb = maxrss_bytes / (1024 ** 2)
# Create a 3D scatter plot for MaxRSS with human-readable format
fig = go.Figure(data=[go.Scatter3d(
    x=x_data,
    y=y_data,
    z=z_data,
    mode='markers',
    customdata=custom_data,  # Include 'MaxRSS' in customdata
    marker=dict(
        size=5,
        color=maxrss_mb,             # Set color to the MaxRSS values in MB
        colorscale=[[0, 'green'], [1, 'red']],  # Custom color scale: green to red
        colorbar=dict(title="MaxRSS (MB)"),     # Show MB in the color bar
        opacity=0.8,
        cmin=0,                       # Set color scale minimum to 0
        cmax=max(maxrss_mb)           # Set color scale maximum to the max of MaxRSS in MB
    ),
    hovertemplate=(
        'TRAJECTORIES: %{x}<br>'
        'Horizon: %{y}<br>'
        'PARTICLES: %{z}<br>'
        'MaxRSS (formatted): %{customdata[0]}<br>'   # Show 'MaxRSS' from the custom data
        'MaxRSS (MB): %{marker.color:.2f} MB<br>'
        'MaxRSS (human-readable): %{customdata[0]}<br>'  # Human-readable MaxRSS
        'JobID: %{customdata[1]}<br>'             # Show 'JobID' from custom data
        '<extra></extra>'
    )
)])

# Update plot layout
fig.update_layout(
    title='3D Scatter Plot of TRAJECTORIES vs Horizon vs PARTICLES (MaxRSS)',
    scene=dict(
        xaxis_title='TRAJECTORIES',
        yaxis_title='Horizon',
        zaxis_title='PARTICLES'
    )
)

# Show the plot
fig.show()

import plotly.graph_objects as go
import pandas as pd
import numpy as np

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

# Function to convert MaxRSS to bytes
def parse_maxrss(rss_str):
    if pd.isnull(rss_str):
        return np.nan
    rss_str = str(rss_str).strip()
    units = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    for unit in units:
        if rss_str.upper().endswith(unit):
            try:
                return float(rss_str[:-1]) * units[unit]
            except ValueError:
                return np.nan
    try:
        return float(rss_str)
    except ValueError:
        return np.nan

merged_df['MaxRSS_Bytes'] = merged_df['MaxRSS'].apply(parse_maxrss)

# Assuming 'FolderSize' is available in the merged_df


# Drop rows with missing values in key columns
merged_df.dropna(subset=['TRAJECTORIES', 'Horizon', 'PARTICLES', 'ElapsedRaw', 'MaxRSS_Bytes', 'FolderSize'], inplace=True)

# Prepare 3D scatter data
x_data = merged_df['TRAJECTORIES'].values
y_data = merged_df['Horizon'].values
z_data = merged_df['PARTICLES'].values
elapsed_time = merged_df['ElapsedRaw'].values
# Create a 3D scatter plot with ElapsedRaw as the color scale and hover info
# Prepare custom data to include the 'Elapsed' column for hover info
custom_data = merged_df[['Elapsed']].values

# Create a 3D scatter plot with ElapsedRaw and Elapsed in hover info
fig = go.Figure(data=[go.Scatter3d(
    x=x_data,
    y=y_data,
    z=z_data,
    mode='markers',
    customdata=custom_data,  # Include 'Elapsed' column in customdata
    marker=dict(
        size=5,
        color=elapsed_time,           # set color to the elapsed time values
        colorscale=[[0, 'green'], [1, 'red']],  # Custom color scale: green to red
        colorbar=dict(title="Elapsed Time (seconds)"),  # No 'K' in the title
        opacity=0.8
    ),
    hovertemplate=(
        'TRAJECTORIES: %{x}<br>'
        'Horizon: %{y}<br>'
        'PARTICLES: %{z}<br>'
        'Elapsed Time (s): %{marker.color:.2f}<br>'
        'Elapsed (formatted): %{customdata[0]}<extra></extra>'  # Show 'Elapsed' from the custom data
    )
)])
# Update plot layout
fig.update_layout(
    title='3D Scatter Plot of TRAJECTORIES vs Horizon vs PARTICLES (elapsed time)',
    scene=dict(
        xaxis_title='TRAJECTORIES',
        yaxis_title='Horizon',
        zaxis_title='PARTICLES'
    )
)

# Show the plot
fig.show()

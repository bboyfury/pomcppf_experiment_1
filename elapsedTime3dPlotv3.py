import plotly.graph_objects as go
import pandas as pd

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

# Function to convert MaxRSS to bytes
def parse_maxrss(value):
    if isinstance(value, str) and value.endswith('K'):
        return float(value[:-1]) * 1024
    elif isinstance(value, str) and value.endswith('M'):
        return float(value[:-1]) * 1024 * 1024
    elif isinstance(value, str) and value.endswith('G'):
        return float(value[:-1]) * 1024 * 1024 * 1024
    else:
        return float(value)

# Convert necessary columns to numeric types
numeric_columns = ['TRAJECTORIES', 'Horizon', 'PARTICLES', 'ElapsedRaw']
for col in numeric_columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Convert elapsed time to minutes
merged_df['ElapsedRaw'] = merged_df['ElapsedRaw'] / 60  # Convert seconds to minutes

# Drop rows with missing values in key columns
merged_df.dropna(subset=['TRAJECTORIES', 'Horizon', 'ElapsedRaw', 'PARTICLES'], inplace=True)

# Scale PARTICLES for marker size (adjust scaling factor for visual clarity)
particle_sizes = merged_df['PARTICLES'] / merged_df['PARTICLES'].max() * 35  # Smaller particle sizes

# Create the 3D scatter plot
fig = go.Figure()

# Add scatter plot
fig.add_trace(go.Scatter3d(
    x=merged_df['Horizon'],
    y=merged_df['TRAJECTORIES'],
    z=merged_df['ElapsedRaw'],
    mode='markers',
    marker=dict(
        size=particle_sizes,
        color=merged_df['ElapsedRaw'],
        colorscale='RdYlGn_r',
        colorbar=dict(title="Elapsed Time (minutes)"),
        showscale=True,
    ),
    hovertemplate=(
        'Horizon: %{x}<br>'
        'TRAJECTORIES: %{y}<br>'
        'Elapsed Time (minutes): %{z}<br>'
        'PARTICLES: %{text}'
    ),
    text=merged_df['PARTICLES'],  # Clear PARTICLES information in hover
))

fig.update_layout(
    scene=dict(
        xaxis_title='Horizon',
        yaxis_title='TRAJECTORIES',
        zaxis_title='Elapsed Time (minutes)'
    ),
    title="3D Scatter Plot of Horizon vs TRAJECTORIES vs Elapsed Time",
    legend_title="Particle Size",
    annotations=[dict(
        text="Bigger spheres represent higher particle count",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=12)
    )]
)

# Show the plot
fig.show()

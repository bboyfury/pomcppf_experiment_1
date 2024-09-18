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

# Apply the function to convert MaxRSS to bytes
merged_df['MaxRSS_Bytes'] = merged_df['MaxRSS'].apply(parse_maxrss)

# Convert MaxRSS_Bytes to MB (Megabytes)
merged_df['MaxRSS_MB'] = merged_df['MaxRSS_Bytes'] / (1024 * 1024)

# Drop rows with missing values in key columns
merged_df.dropna(subset=['TRAJECTORIES', 'Horizon', 'MaxRSS_MB', 'PARTICLES'], inplace=True)

# Scale PARTICLES for smaller marker sizes (adjust scaling factor for visual clarity)
particle_sizes = merged_df['PARTICLES'] / merged_df['PARTICLES'].max() * 35  # Smaller size scaling

# Create a 3D scatter plot using Plotly
fig = go.Figure()

# Scatter plot with color mapping (red for high, green for low) and varying marker sizes for PARTICLES
fig.add_trace(go.Scatter3d(
    x=merged_df['Horizon'],
    y=merged_df['TRAJECTORIES'],
    z=merged_df['MaxRSS_MB'],
    mode='markers',
    marker=dict(
        size=particle_sizes,
        color=merged_df['MaxRSS_MB'],  # Color by MaxRSS
        colorscale='RdYlGn_r',  # Colorscale reversed
        colorbar=dict(title='MaxRSS (MB)'),
        showscale=True
    ),
    hovertemplate=
        'Horizon: %{x}<br>' +
        'TRAJECTORIES: %{y}<br>' +
        'MaxRSS (MB): %{z}<br>' +
        'PARTICLES: %{marker.size:.2f}<extra></extra>'  # Shows particles info on hover
))

# Update the layout with axis labels
fig.update_layout(
    scene=dict(
        xaxis_title='Horizon',
        yaxis_title='TRAJECTORIES',
        zaxis_title='MaxRSS (MB)'
    ),
    title='3D Scatter Plot of Horizon, TRAJECTORIES, and MaxRSS',
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

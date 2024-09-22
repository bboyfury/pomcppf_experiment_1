import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the data
df = pd.read_csv('finalresults_with_rewards.csv')

# Sort the data from min to max AverageReward
df_sorted = df.sort_values(by='AverageReward', ascending=True)

# Extract every 17th row from the sorted data
extracted_rows = df_sorted.iloc[::17].reset_index(drop=True)

# Create a configuration tuple as string
extracted_rows['Config'] = extracted_rows.apply(lambda row: f"({row['TRAJECTORIES']}, {row['Horizon']}, {row['PARTICLES']})", axis=1)

# Define colors
trajectory_color = 'purple'
horizon_color = 'green'
particles_color = 'blue'

# Define bar width
bar_width = 0.4  # Wider bars to make space for labels

# Set up the plot
plt.figure(figsize=(16, 10))

# Positions for each configuration
y_positions = range(len(extracted_rows))

# Plot bars representing rewards
plt.barh(y_positions, extracted_rows['AverageReward'], height=bar_width, color='lightgray')

# Set y-ticks to configurations
plt.yticks(y_positions, extracted_rows['Config'])

# Set labels and title
plt.xlabel('Average Reward')
plt.ylabel('Configuration (Trajectory, Horizon, Particles)')
plt.title('Average Reward by Every 17th Configuration')

# Annotate bars with parameter values
for i, row in extracted_rows.iterrows():
    # Trajectory
    plt.text(row['AverageReward'] - 0.5, i - bar_width/3, f"T: {row['TRAJECTORIES']}",
             va='center', color=trajectory_color, fontsize=10, fontweight='bold')
    # Horizon
    plt.text(row['AverageReward'] - 0.5, i, f"H: {row['Horizon']}",
             va='center', color=horizon_color, fontsize=10, fontweight='bold')
    # Particles
    plt.text(row['AverageReward'] - 0.5, i + bar_width/3, f"P: {row['PARTICLES']}",
             va='center', color=particles_color, fontsize=10, fontweight='bold')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

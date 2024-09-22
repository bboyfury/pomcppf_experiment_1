import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# -------------------- Step 1: Read and Sort the Data -------------------- #

# Define the path to your CSV file
csv_file = 'sorted_finalresults_with_rewards.csv'  # Replace with your actual file path if different

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(csv_file)
    print(f"Successfully read '{csv_file}'.")
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file '{csv_file}' is empty.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: The file '{csv_file}' does not appear to be in CSV format.")
    exit(1)

# Verify required columns exist
required_columns = {'MainJobID', 'ElapsedRaw', 'MaxRSS', 'TRAJECTORIES', 'Horizon', 'PARTICLES', 'AverageReward'}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    print(f"Error: The following required columns are missing from the CSV: {missing}")
    exit(1)

# Sort the DataFrame by 'AverageReward' in ascending order
df_sorted = df.sort_values(by='AverageReward', ascending=True).reset_index(drop=True)
print("DataFrame sorted by 'AverageReward' in ascending order.")

# -------------------- Step 2: Select Specific Rows Including the Last Row -------------------- #

def select_rows_with_last(df, step):
    """
    Selects every 'step' row from the DataFrame and ensures the last row is included.

    Parameters:
    - df: pandas DataFrame
    - step: int, step size (e.g., 10 for every 10th row)

    Returns:
    - selected_df: pandas DataFrame with selected rows
    """
    if step <= 0:
        raise ValueError("Step size must be a positive integer.")
    
    # Select every 'step' row starting from the first row (index 0)
    selected_df = df.iloc[::step].copy()
    print(f"Selected every {step}th row.")
    
    # Check if the last row is already included
    if (len(df) - 1) % step != 0:
        # Convert the last row to a DataFrame
        last_row = df.iloc[-1:]
        # Concatenate the last row to selected_df
        selected_df = pd.concat([selected_df, last_row], ignore_index=True)
        print("Appended the last row to the selected rows.")
    else:
        print("The last row is already included in the selected rows.")
    
    return selected_df

# Define the step size (e.g., every 10th row)
step_size = 10

# Select the rows
selected_rows = select_rows_with_last(df_sorted, step_size).reset_index(drop=True)

# Number of selected rows
num_selected = len(selected_rows)
print(f"Total selected rows for the bar chart: {num_selected}\n")

# Display selected rows (optional)
print("Selected Rows:")
print(selected_rows[['MainJobID', 'AverageReward', 'TRAJECTORIES', 'Horizon', 'PARTICLES']])
print("\n")

# -------------------- Step 3: Plotting the Bar Chart -------------------- #

# Set up the bar chart
fig, ax = plt.subplots(figsize=(20, 10))  # Adjust figure size as needed

# Define bar width
bar_width = 0.2

# Generate indices for the selected rows
indices = np.arange(num_selected)

# Heights for all bars are the 'AverageReward'
heights = selected_rows['AverageReward']

# Plot three bars for each selected row
bars1 = ax.bar(indices - bar_width, heights, 
               width=bar_width, color='purple', label='TRAJECTORIES')

bars2 = ax.bar(indices, heights, 
               width=bar_width, color='green', label='Horizon')

bars3 = ax.bar(indices + bar_width, heights, 
               width=bar_width, color='yellow', label='PARTICLES')

# Set x-axis labels
ax.set_xlabel('Jobs from min to max(sum of Reward of all runs for each conf / number of runs(100) )', fontsize=16)
ax.set_ylabel('Average Reward', fontsize=16)
ax.set_title('Agent Reward Based on TRAJECTORIES, Horizon, and PARTICLES - setup 1, online, POMCPPF', fontsize=20)

# Customize x-ticks
ax.set_xticks(indices)
ax.set_xticklabels([f'Job {i * step_size + 1}' for i in indices], rotation=45, ha='right', fontsize=12)

# Create custom legend patches with matching colors
trajectory_patch = mpatches.Patch(color='purple', label='TRAJECTORIES')
horizon_patch = mpatches.Patch(color='green', label='Horizon')
particle_patch = mpatches.Patch(color='yellow', label='PARTICLES')
ax.legend(handles=[trajectory_patch, horizon_patch, particle_patch], fontsize=14)

# Add annotations for each bar (parameter values in the middle with different colors)
def autolabel_middle(bars, values, annotation_color):
    """Attach a text label in the middle of each bar displaying its value."""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value}',
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0),  # No offset
                    textcoords="offset points",
                    ha='center', va='center', fontsize=10, color=annotation_color, fontweight='bold')

# Define annotation colors different from bar colors for better readability
annotation_colors = {
    'TRAJECTORIES': 'black',
    'Horizon': 'black',
    'PARTICLES': 'black'
}

# Add annotations for each bar (parameter values in the middle with different colors)
def autolabel_middle_vertical(bars, values, annotation_color):
    """Attach a text label vertically at the center of each bar displaying its value."""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value}',
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0),  # No offset
                    textcoords="offset points",
                    ha='center', va='center', fontsize=10, color=annotation_color, fontweight='bold', rotation=90)  # 90-degree rotation for vertical display

# Annotate each set of bars with their respective values vertically
autolabel_middle_vertical(bars1, selected_rows['TRAJECTORIES'], annotation_colors['TRAJECTORIES'])
autolabel_middle_vertical(bars2, selected_rows['Horizon'], annotation_colors['Horizon'])
autolabel_middle_vertical(bars3, selected_rows['PARTICLES'], annotation_colors['PARTICLES'])


# Remove 'AverageReward' annotations above the bars by commenting out or deleting the following block
# for i, reward in enumerate(heights):
#     ax.text(indices[i], reward + max(heights)*0.01, 
#             f'{reward}', 
#             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Set y-axis limit slightly higher than the max 'AverageReward' for annotations
ax.set_ylim(0, max(heights) * 1.3)

# Optional: Add gridlines for better readability
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

# Ensure layout fits well
plt.tight_layout()

# Display the plot
plt.savefig('setup1.png')
plt.show()

# -------------------- Step 4: Verify Inclusion of the Last Row -------------------- #

# Retrieve the last row using iloc
last_row = df_sorted.iloc[-1]

# Check if the last row is in selected_rows
is_last_row_included = selected_rows.iloc[-1].equals(last_row)
print(f"Is the last row included in selected_rows? {is_last_row_included}")

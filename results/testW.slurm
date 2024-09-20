#!/bin/bash
#SBATCH --job-name=wildfire_pomcppf_test
#SBATCH --output=logs/wildfire_pomcppf_test_job_%j.out
#SBATCH --error=logs/wildfire_pomcppf_test_job_%j.err
#SBATCH --time=01:00:00          # Reduced time for testing
#SBATCH --mem=4G                 # Allocated memory for testing
#SBATCH --cpus-per-task=1        # Number of CPU cores per task for testing
#SBATCH --array=0-1              # Number of test combinations (e.g., 2)
#SBATCH --begin=now
#SBATCH --mail-type=END         # Send email upon job completion
#SBATCH --mail-user=alirezasalehabadi.as@gmail.com    # Notification email address

# Ensure the logs and outputs directories exist
mkdir -p logs
mkdir -p outputs

# Load necessary modules (modify if using a virtual environment or specific modules)
module load python/3.8  # Replace with your required Python version

# Define parameter arrays for testing
trajectories=(2)          # Fixed trajectory value for testing
horizons=(2)              # Fixed horizon value for testing
ucbs=(0 0.5)              # Limited UCB values for testing
particles=(2)             # Fixed particle value for testing

# Calculate indices for parameter combinations
total_trajectories=${#trajectories[@]}
total_horizons=${#horizons[@]}
total_ucbs=${#ucbs[@]}
total_particles=${#particles[@]}

# Calculate total combinations
total_combinations=$((total_trajectories * total_horizons * total_ucbs * total_particles))

# Verify array index is within range
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) out of range (0-$((total_combinations-1)))."
  exit 1
fi

# Determine the combination based on the array task ID
index=$SLURM_ARRAY_TASK_ID

traj_index=$(( index / (total_horizons * total_ucbs * total_particles) ))
remainder=$(( index % (total_horizons * total_ucbs * total_particles) ))

horizon_index=$(( remainder / (total_ucbs * total_particles) ))
remainder=$(( remainder % (total_ucbs * total_particles) ))

ucb_index=$(( remainder / total_particles ))
particle_index=$(( remainder % total_particles ))

# Retrieve parameter values
traj=${trajectories[$traj_index]}
horizon=${horizons[$horizon_index]}
ucb=${ucbs[$ucb_index]}
particle=${particles[$particle_index]}

# Define output file name based on parameters
output_file="outputs/output_traj${traj}_horizon${horizon}_ucb${ucb}_particles${particle}.txt"

echo "======================================"
echo "Running combination:"
echo "Trajectories: $traj"
echo "Horizon: $horizon"
echo "UCB: $ucb"
echo "Particles: $particle"
echo "Output will be saved to: $output_file"
echo "======================================"

# Execute the Python script and capture execution time
{
  echo "=== Script Output ==="
  python generate_wildfire_pompcppf_policy.py 1 0.0 "$traj" "$horizon" "$ucb" "$particle" 1 1
  echo ""
  echo "=== Execution Time ==="
  /usr/bin/time -f "Elapsed Time: %E" python generate_wildfire_pompcppf_policy.py 1 0.0 "$traj" "$horizon" "$ucb" "$particle" 1 1 > /dev/null
} &> "$output_file"

echo "======================================"
echo "Completed combination:"
echo "Trajectories: $traj, Horizon: $horizon, UCB: $ucb, Particles: $particle"
echo "Execution time recorded in: $output_file"
echo "======================================"

#!/bin/bash
#SBATCH --job-name=wildfire_pomcppf
#SBATCH --output=logs/wildfire_pomcppf_job_%j.out
#SBATCH --error=logs/wildfire_pomcppf_job_%j.err
#SBATCH --time=04:00:00          # Adjust based on expected runtime
#SBATCH --mem=0                # Adjust based on memory requirements
#SBATCH --cpus-per-task=1        # Number of CPU cores per task
#SBATCH --array=0-81            # Total combinations (3 traj * 3 horizon * 3 ucb * 3 particles = 144)
#SBATCH --begin=now
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=alirezasalehabadi.as@gmail.com    # Email address for notifications
# Ensure the logs directory exists

#SBATCH --nodes=1           # Each job gets 1 node
#SBATCH --ntasks=1          # 1 task per job
#SBATCH --cpus-per-task=4   # Number of CPUs per task

mkdir -p logs
mkdir -p outputs

# Load necessary modules (modify if using a virtual environment or specific modules)
module load python/3.8  # Replace with your Python version

# Define parameter arrays
trajectories=(5 50 200)
horizons=(5 50 200)
ucbs=(0 0.5 1)
particles=(5 10 20)

# Calculate indices for parameter combinations
total_trajectories=${#trajectories[@]}
total_horizons=${#horizons[@]}
total_ucbs=${#ucbs[@]}
total_particles=${#particles[@]}

# Calculate total combinations
total_combinations=$((total_trajectories * total_horizons * total_ucbs * total_particles))

# Verify array index
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID out of range."
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

echo "Running combination:"
echo "Trajectories: $traj, Horizon: $horizon, UCB: $ucb, Particles: $particle"
echo "Output will be saved to: $output_file"

# Execute the Python script and capture execution time
{
  echo "=== Output ==="
  python generate_wildfire_pompcppf_policy.py 1 0.0 "$traj" "$horizon" "$ucb" "$particle" 1 1
  echo ""
  echo "=== Execution Time ==="
  /usr/bin/time -f "Elapsed Time: %E" python generate_wildfire_pompcppf_policy.py 1 0.0 "$traj" "$horizon" "$ucb" "$particle" 1 1 > /dev/null
} &> "$output_file"

echo "Completed combination:"
echo "Trajectories: $traj, Horizon: $horizon, UCB: $ucb, Particles: $particle"
echo "Execution time recorded in: $output_file"
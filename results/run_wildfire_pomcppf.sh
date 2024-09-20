#!/bin/bash

# =============================================================================
# Script to execute generate_wildfire_pompcppf_policy.py with different
# combinations of trajectories, horizons, and particles.
# Outputs and execution times are saved for each combination.
# =============================================================================

# Create necessary directories for logs and outputs
mkdir -p logs
mkdir -p outputs

# Load necessary modules (modify as needed)
module load python/3.8  # Replace with your Python version

# Define parameter arrays
trajectories=(5 50 200)
horizons=(5 50 200)
particles=(5 10 20)

# Total number of combinations
total_combinations=$(( ${#trajectories[@]} * ${#horizons[@]} * ${#particles[@]} ))

echo "Starting execution of $total_combinations combinations."

# Iterate through all combinations of trajectories, horizons, and particles
for traj in "${trajectories[@]}"; do
  for horizon in "${horizons[@]}"; do
    for particle in "${particles[@]}"; do
      
      # Define the output file name based on current parameters
      output_file="outputs/output_traj${traj}_horizon${horizon}_particles${particle}.txt"
      
      echo "------------------------------------------------------------"
      echo "Running combination:"
      echo "Trajectories: $traj, Horizon: $horizon,ucb: 1, Particles: $particle"
      echo "Output will be saved to: $output_file"
      echo "------------------------------------------------------------"
      
      # Execute the Python script and capture output and execution time
      {
        echo "=== Output ==="
        python generate_wildfire_pompcppf_policy.py 1 0.0 "$traj" "$horizon" 1 "$particle" 1 1
        echo ""
        echo "=== Execution Time ==="
        /usr/bin/time -f "Elapsed Time: %E" python generate_wildfire_pompcppf_policy.py 1 0.0 "$traj" 1 "$horizon" "$particle" 1 1
      } &> "$output_file"
      
      echo "Completed combination:"
      echo "Trajectories: $traj, Horizon: $horizon, Particles: $particle"
      echo "Execution time recorded in: $output_file"
      echo ""
      
    done
  done
done

echo "All combinations have been executed."
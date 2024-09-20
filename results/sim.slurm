#!/bin/bash
#SBATCH --job-name=wildfire_simulation
#SBATCH --output=wildfire_simulation_%A_%a.out
#SBATCH --error=wildfire_simulation_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-144

# Define parameter arrays
horizons=(5 50 100 500)
trajectories=(5 50 100 500)
ucbs=(0 0.5 1)
particles=(5 10 20)

# Calculate indices for array job
total_h=${#horizons[@]}
total_t=${#trajectories[@]}
total_u=${#ucbs[@]}
total_p=${#particles[@]}

let "h_total = total_h"
let "t_total = total_t"
let "u_total = total_u"
let "p_total = total_p"

let "h_size = t_total * u_total * p_total"
let "t_size = u_total * p_total"
let "u_size = p_total"

# Get array job index (0-based)
index=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate parameter indices
h_index=$(( index / h_size ))
remainder=$(( index % h_size ))
t_index=$(( remainder / u_size ))
remainder=$(( remainder % u_size ))
u_index=$(( remainder / p_total ))
p_index=$(( remainder % p_total ))

# Get parameter values
horizon=${horizons[$h_index]}
trajectory=${trajectories[$t_index]}
ucb=${ucbs[$u_index]}
particle=${particles[$p_index]}

# Define other fixed parameters
setup="default_setup"
start_run=1
stop_run=10
planning_level=1
offline_policy=1

# Define output filename
output_file="trajectory_${trajectory}_horizon_${horizon}_ucb_${ucb}_particles_${particle}.txt"

# Execute the command and measure time
/usr/bin/time -f "Elapsed time: %E" \
    python run_wildfire_ipomcppf_simulation.py \
    "$setup" "$start_run" "$stop_run" 0.0 \
    "$trajectory" "$horizon" "$ucb" "$planning_level" "$particle" \
    "$offline_policy" > "${output_file}" 2>&1
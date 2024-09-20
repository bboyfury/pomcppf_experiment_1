#!/bin/bash
#SBATCH --job-name=wf-pomcppf-${SLURM_JOB_ID}
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out

module load python/3.9

OUTPUT_DIR="${SLURM_SUBMIT_DIR}/output_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Change to the output directory
cd "$OUTPUT_DIR"

# Print some information (optional)
echo "Job ID: $SLURM_JOB_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "Running on host: $(hostname)"
echo "Time is: $(date)"

# Run your application or script
# For example, if you have a Python script:
# python "${SLURM_SUBMIT_DIR}/my_script.py" > my_script_output.txt 2>&1

# Placeholder command (replace with your actual command)
echo "Hello from SLURM job $SLURM_JOB_ID SETUP=1 START_RUN=1 STOP_RUN=100 EPSILON=0.0 TRAJECTORIES=100 HORIZON=4 UCB_C=50 PARTICLES=50" > job_output_${SLURM_JOB_ID}.txt

source /work/soh/alirezas/run1/OASYS/scripts/wildfire/myenv/bin/activate

python /work/soh/alirezas/run1/OASYS/scripts/wildfire/run_wildfire_pomcppf_simulation.py 1 1 1 0.0 1 4 10 10


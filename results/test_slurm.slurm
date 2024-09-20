#!/bin/bash
#SBATCH --job-name=wf-pomcppf-${SLURM_JOB_ID}
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=100
#SBATCH --error=%x_%j.err
#SBATCH --output=%x_%j.out

#SBATCH --mail-type=END,FAIL         # Send email upon job completion
#SBATCH --mail-user=alirezasalehabadi.as@gmail.com    # Notification email address


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
#echo "Hello from SLURM job $SLURM_JOB_ID SETUP=1 START_RUN=1 STOP_RUN=100 EPSILON=0.0 TRAJECTORIES=100 HORIZON=4 UCB_C=50 PARTICLES=50" > job_output_${SLURM_JOB_ID}.txt

source /work/soh/alirezas/run1/OASYS/myenv/bin/activate

python /work/soh/alirezas/run1/OASYS/scripts/wildfire/run_wildfire_pomcppf_simulation.py 1 1 5 0.0 5 4 80 5 0
JOB_STATS_OUTPUT="${SLURM_SUBMIT_DIR}/job_statistics_${SETUP}_${START_RUN}_${STOP_RUN}_${EPSILON}_${TRAJECTORIES}_${HORIZON}_${UCB_C}_${PARTICLES}.csv"
JOB_STATS=$(sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize -P | tail -n 1) 

echo  "z Job ID: $JOB_ID\nElapsed Time: $ELAPSED_TIME\nMax RSS: $MAX_RSS\nMax VMSize: $MAX_VMSIZE" > job_output_${SLURM_JOB_ID}.txt
echo  "x $JOB_STATS" > job_output_${SLURM_JOB_ID}.txt
echo  "y $JOB_STATS_OUTPUT" > job_output_${SLURM_JOB_ID}.txt


# Append the job statistics along with parameters to a CSV file
echo "JobID,Trajectories,Horizon,Particles,Elapsed,MaxRSS,MaxVMSize" > $JOB_STATS_OUTPUT  # Write header only if the file does not exist
echo "$SLURM_JOB_ID,$TRAJECTORIES,$HORIZON,$PARTICLES,$JOB_STATS" >> $JOB_STATS_OUTPUT
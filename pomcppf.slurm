#!/bin/bash
#SBATCH --job-name=wf-pomcppf-${SLURM_JOB_ID}
#SBATCH --time=160:00:00
#SBATCH --mem-per-cpu=130000
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

# Check if the correct number of arguments is provided
if [ "$#" -ne 8 ]; then
  echo "Usage: sbatch $0 SETUP START_RUN STOP_RUN EPSILON TRAJECTORIES HORIZON UCB_C PARTICLES"
  exit 1
fi

# Assign command-line arguments to variables
SETUP=$1
START_RUN=$2
STOP_RUN=$3
EPSILON=$4
TRAJECTORIES=$5
HORIZON=$6
UCB_C=$7
PARTICLES=$8

# Print the command for debugging (optional)
echo "Running with parameters: SETUP=$SETUP START_RUN=$START_RUN STOP_RUN=$STOP_RUN EPSILON=$EPSILON TRAJECTORIES=$TRAJECTORIES HORIZON=$HORIZON UCB_C=$UCB_C PARTICLES"

# Activate your environment
source /work/soh/alirezas/run1/OASYS/myenv/bin/activate

# Run your Python script with arguments
python /work/soh/alirezas/run1/OASYS/scripts/wildfire/run_wildfire_pomcppf_simulation.py $SETUP $START_RUN $STOP_RUN $EPSILON $TRAJECTORIES $HORIZON $UCB_C $PARTICLES 0

# Step 4: Collect job statistics using sacct after the job completes
JOB_STATS_OUTPUT="${SLURM_SUBMIT_DIR}/job_statistics_{$SETUP}_$START_RUN}_{$STOP_RUN}_{$EPSILON}_{$TRAJECTORIES}_{$HORIZON}_{$UCB_C}_{$PARTICLES}.csv"
JOB_STATS=$(sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize -P | tail -n 1)

# Append the job statistics along with parameters to a CSV file
echo "JobID,Trajectories,Horizon,Particles,Elapsed,MaxRSS,MaxVMSize" > $JOB_STATS_OUTPUT  # Write header only if the file does not exist
echo "$SLURM_JOB_ID,$TRAJECTORIES,$HORIZON,$PARTICLES,$JOB_STATS" >> $JOB_STATS_OUTPUT

echo "Job statistics saved to $JOB_STATS_OUTPUT."

JOB_STATS_OUTPUT="${SLURM_SUBMIT_DIR}/job_statistics_${SETUP}_${START_RUN}_${STOP_RUN}_${EPSILON}_${TRAJECTORIES}_${HORIZON}_${UCB_C}_${PARTICLES}.csv"
JOB_STATS=$(sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize -P | tail -n 1)

echo  "z Job ID: $JOB_ID\nElapsed Time: $ELAPSED_TIME\nMax RSS: $MAX_RSS\nMax VMSize: $MAX_VMSIZE" > job_output_${SLURM_JOB_ID}.txt
echo  "x $JOB_STATS" > job_output_${SLURM_JOB_ID}.txt
echo  "y $JOB_STATS_OUTPUT" > job_output_${SLURM_JOB_ID}.txt

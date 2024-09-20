#!/bin/bash

# Constant parameters
SETUP=1
START_RUN=1
STOP_RUN=100
EPSILON=0.0
UCB_C=25

# Arrays of variable parameters
TRAJECTORIES_LIST=(10 50 100 200 300 500 1000)
HORIZON_LIST=(5 10 20 30)
PARTICLES_LIST=(5 10 20 30 50 100 200)

# Loop over each combination of trajectories, horizon, and particles
for TRAJECTORIES in "${TRAJECTORIES_LIST[@]}"; do
  for HORIZON in "${HORIZON_LIST[@]}"; do
    for PARTICLES in "${PARTICLES_LIST[@]}"; do
      # Submit the SLURM job with the current set of parameters
      echo "Submitting job with TRAJECTORIES=$TRAJECTORIES, HORIZON=$HORIZON, PARTICLES=$PARTICLES"
      sbatch pomcppf.slurm $SETUP $START_RUN $STOP_RUN $EPSILON $TRAJECTORIES $HORIZON $UCB_C $PARTICLES
    done
  done
done

echo "All jobs submitted."

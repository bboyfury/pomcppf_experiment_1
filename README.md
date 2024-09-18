POMCP Experiment Report
Overview

This repository contains the results of my POMCP experiments, focusing on memory usage and execution time for various parameter combinations. The experiments were conducted using Slurm to manage job submissions for different configurations of trajectories, horizons, and particles.
Contents

    CSV Files: Contain detailed records of job parameters and results (memory usage and execution time).
    3D Dot Plots: Visualizations showing the impact of different parameter combinations on memory usage and execution time.

Experiment Details

    Constants:
        SETUP=1
        START_RUN=1
        STOP_RUN=100
        EPSILON=0.0
        UCB_C=25

    Variable Parameters:
        TRAJECTORIES_LIST: (10, 50, 100, 200, 300, 500, 1000)
        HORIZON_LIST: (5, 10, 20, 30)
        PARTICLES_LIST: (5, 10, 20, 30, 50, 100, 200)

This results in 196 combinations, which were submitted as individual jobs via Slurm.
Notes

    There are currently 2 jobs still running. Once those are completed, the results will be added to this report.
    Two 3D dot plots are provided to illustrate the relationship between parameter combinations and memory usage/execution time.

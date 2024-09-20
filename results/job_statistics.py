from io import StringIO
import os
import glob
import subprocess
import pandas as pd
import re


def GetStatisticFromCSVFiles():
    # Step 1: Get list of CSV files starting with 'job_statistics_' and ending with '.csv'
    csv_files = glob.glob('job_statistics_*.csv')

    # List to store JobIDs and extracted numbers from filenames
    job_info = []

    # Step 2: Loop through each file
    for file in csv_files:
        # Extract numbers inside curly braces {} from the filename
        numbers_in_braces = re.findall(r'\{(.*?)\}', file)

        # Open the CSV file using pandas
        df = pd.read_csv(file)

        # Step 3: Extract the JobID from the 'JobID' column
        # Assuming you want the first JobID in each file
        job_id = df['JobID'].iloc[0]
        while len(numbers_in_braces) < 8:
            # Insert '1' for START_RUN if missing
            numbers_in_braces.insert(1, '1')
        # Ensure extracted numbers match the expected number of fields for the CSV format
        if len(numbers_in_braces) == 8:  # There are 7 fields apart from JobID
            # Add JobID and extracted numbers to the list
            job_info.append([job_id] + numbers_in_braces+[get_directory_size(job_id)])
    # I had a typo in extracting statistics, for start run I put them inside _startrun}. but it is static 1 for my runs.
    # Convert list to DataFrame
    columns = ['JobID', 'SETUP', 'START_RUN', 'START_RUN',
               'EPSILON', 'TRAJECTORIES', 'Horizon', 'UCB_C', 'PARTICLES','FolderSize']
    job_info_df = pd.DataFrame(job_info, columns=columns)

    # Save DataFrame to CSV
    job_info_df.to_csv('job_info_output.csv', index=False)

    print("Job information has been saved to 'job_info_output.csv'.")
    return job_info_df




import os

def get_directory_size(Idstring):
    dir_name = f"output_{Idstring}"
    if os.path.isdir(dir_name):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(dir_name):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Skip if it's a symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    else:
        return 0


def extract_job_statistics(job_info_df):
    desired_fields = "JobID,JobName,Account,AveCPU,AveRSS,AveVMSize,CPUTime,Elapsed,ElapsedRaw,ExitCode,MaxRSS,MaxVMSize," \
                 "MaxDiskRead,MaxDiskReadNode,MaxDiskReadTask,MaxDiskWrite,MaxDiskWriteNode,MaxDiskWriteTask,MaxPages,ReqCPUS,ReqMem,ReqNodes,NCPUS,NNodes,User,Eligible,Start,End,Suspended"
    # Create a list to store job statistics
    job_statistics = []

    # Loop through each job ID from the DataFrame
    for job_id in job_info_df['JobID']:
        job_id_str = str(job_id)
        # Run the 'sacct' command with desired fields for the specific job ID
        try:
            result = subprocess.run(
                ['sacct', '-j', job_id_str, '--format', desired_fields, '--noheader', '--parsable2'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

            # Split the result output by lines and process each line
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                # Split the line by '|' to get each field value
                job_stat = line.split('|')
                # Append the job statistics to the list
                job_statistics.append(job_stat)

        except subprocess.CalledProcessError as e:
            print(f"Error retrieving data for JobID {job_id_str}: {e.stderr}")

    # Convert the list to a DataFrame with the desired columns
    columns = desired_fields.split(',')
    job_statistics_df = pd.DataFrame(job_statistics, columns=columns)

    # Save the DataFrame to a CSV file
    output_file = 'job_statistics_output.csv'
    job_statistics_df.to_csv(output_file, index=False)
    print(f"Job statistics have been saved to '{output_file}'.")

    return job_statistics_df

job_info_df = GetStatisticFromCSVFiles()

job_statistics_df = extract_job_statistics(job_info_df)
#!/bin/bash

# Ensure exactly 8 arguments are provided
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8"
    echo "Example: $0 1 0.0 200 5 1 20 1 1"
    exit 1
fi

# Assign command-line arguments to variables
ARG1=$1
ARG2=$2
ARG3=$3
ARG4=$4
ARG5=$5
ARG6=$6
ARG7=$7
ARG8=$8

# Function to sanitize filename components
sanitize() {
    echo "$1" | tr '.' '_' | tr '-' 'n'  # Replace dots with underscores and dashes with 'n'
}

# Sanitize arguments for filename
SAN_ARG1=$(sanitize "$ARG1")
SAN_ARG2=$(sanitize "$ARG2")
SAN_ARG3=$(sanitize "$ARG3")
SAN_ARG4=$(sanitize "$ARG4")
SAN_ARG5=$(sanitize "$ARG5")
SAN_ARG6=$(sanitize "$ARG6")
SAN_ARG7=$(sanitize "$ARG7")
SAN_ARG8=$(sanitize "$ARG8")

# Construct the output filename based on parameters
OUTPUT_DIR="output"
OUTPUT_FILE="${OUTPUT_DIR}/output_${SAN_ARG1}_${SAN_ARG2}_${SAN_ARG3}_${SAN_ARG4}_${SAN_ARG5}_${SAN_ARG6}_${SAN_ARG7}_${SAN_ARG8}.txt"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the Python command
COMMAND=(python generate_wildfire_pompcppf_policy.py "$ARG1" "$ARG2" "$ARG3" "$ARG4" "$ARG5" "$ARG6" "$ARG7" "$ARG8")

# Record the start time with milliseconds precision
START_TIME=$(date +%s.%3N)

# Execute the command and capture both stdout and stderr
"${COMMAND[@]}" > "$OUTPUT_FILE" 2>&1

# Record the end time with milliseconds precision
END_TIME=$(date +%s.%3N)

# Calculate execution time using bc for floating point arithmetic
EXEC_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# Append execution time to the output file
echo -e "\nExecution Time: ${EXEC_TIME} seconds" >> "$OUTPUT_FILE"

echo "Command executed successfully."
echo "Output saved to $OUTPUT_FILE"
 
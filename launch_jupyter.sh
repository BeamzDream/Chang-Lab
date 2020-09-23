#!/bin/bash

jobid=`sbatch ./run_jupyter.sh`

# Pull out just the id
jobid=${jobid/Submitted batch job }

# Wait for the job file to be created and when it is print
while [ ! -e ./jupyter-log-${jobid}.txt ]
do
  sleep 1s
done

# Once the file is created, wait for it to have printed at least 10 lines
line_number=`cat ./jupyter-log-${jobid}.txt | wc -l`
while [ $line_number -lt 15 ]
do
  sleep 1s;
  line_number=`cat ./jupyter-log-${jobid}.txt | wc -l`
done

# Print the script once it exists
cat ./jupyter-log-${jobid}.txt

# Warn user
echo "##### Be aware, job should be terminated with scancel $jobid when you have finished with the notebook ######"

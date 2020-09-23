#!/bin/bash
#
# Run the jupyter notebook that will be ssh'd into
#
#SBATCH --partition interactive
#SBATCH --nodes 1
#SBATCH --time 4:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --job-name tunnel
#SBATCH --output ./jupyter-log-%J.txt

## get tunneling info
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@node.zoo.cs.yale.edu
    -----------------------------------------------------------------

    Then open a browser on your local machine to the following address
    ------------------------------------------------------------------
    localhost:$ipnport  (prefix w/ https:// if using password)
    ------------------------------------------------------------------
    "

## start an ipcluster instance and launch jupyter server
module load mpi/openmpi-x86_64
mpirun -n 1 jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip

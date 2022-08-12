#!/bin/bash

##################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like ##SBATCH
##################

#SBATCH --partition rtx2080			### specify partition name where to run a job. debug: 2 hours limit; short: 7 days limit; gtx1080: 7 days
#SBATCH --time 7-00:00:00			### limit the time of job running, partition limit can override this. Format: D-H:MM:SS
#SBATCH --job-name CP-RT	### name of the job
#SBATCH --output=%a_your_output.out			### output log for running job - %J for job number
#SBATCH --mail-user=leorro@post.bgu.ac.il	### user email for sending job status
#SBATCH --mail-type=ALL			### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

#SBATCH --gpus=1				### number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU
#SBATCH --mem=64G				### amount of RAM memory
#SBATCH --cpus-per-task=6			### number of CPU cores

##SBATCH --array=100-106,110-116,120-126,130-136,140-146,200-206,210-216,220-226,230-236,240-246
##SBATCH --array=101-135,201-235,301-335,401-435,501-535%8
#SBATCH --array=0-4%8

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAYTASKID:\t" $SLURM_ARRAY_TASK_ID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start you code below ####
module load anaconda				### load anaconda module (must present when working with conda environments)
source activate pt-env				### activating environment, environment must be configured before running the job
python /sise/assafzar-group/assafzar/s-and-l/CellProfiling-research/code/learning_tabular_scaled/main.py $SLURM_ARRAY_TASK_ID		### replace with your own command
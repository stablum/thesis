#!/bin/bash
KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
SSH="ssh -i $KEY fstablum@$HOST"
SLEEPTIME=30
ssh-add $KEY
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
bash sync_src.sh

# kill the job before 7 AM
AVAILABLEHOURS=$(expr \( 24 + 6 - $(date +%H) \) % 24)
echo "AVAILABLEHOURS:$AVAILABLEHOURS"
#NODETYPE="ngpus=1"
#NODETYPE="GTX680"
#NODETYPE="K20"
#NODETYPE="C2050"
NODETYPE="GTX480"
echo "NODETYPE:$NODETYPE"
echo "TIMESTAMP:$TIMESTAMP"
echo "after submitting the job will sleep for $SLEEPTIME seconds before retrieving outputs"
$SSH "module load slurm ; quota -m; cd thesis; bash squeue_details.sh | grep fstablum ; sbatch -J '$@' -o $TIMESTAMP.out -e $TIMESTAMP.err -D . --constraint='$NODETYPE' --time='$AVAILABLEHOURS:00:00' job.sh $@ ; sleep $SLEEPTIME; tail -n 100 -f $TIMESTAMP.out & tail -n 100 -f $TIMESTAMP.err "
 

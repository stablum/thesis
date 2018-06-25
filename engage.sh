#!/bin/bash
KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
SSH="ssh -i $KEY fstablum@$HOST"
SLEEPTIME=30
ssh-add $KEY
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
bash sync_src.sh

if test $(date +%u) -eq 5 ; then
    # Friday
    PLUS=48
elif test $(date +%u) -eq 6 ; then
    # Saturday
    PLUS=24
else
    PLUS=0
fi

# kill the job before 8 AM
AVAILABLEHOURS=$(expr \( 24 + 7 - $(date +%H) \) % 24 + $PLUS)
echo "AVAILABLEHOURS:$AVAILABLEHOURS"
#NODETYPE="ngpus=1"
#NODETYPE="GTX680"
#NODETYPE="K20"
#NODETYPE="C2050"
#NODETYPE="GTX480"
#NODETYPE=""
NODETYPE="gpunode"
echo "NODETYPE:$NODETYPE"
echo "TIMESTAMP:$TIMESTAMP"
echo "after submitting the job will sleep for $SLEEPTIME seconds before retrieving outputs"
$SSH "module load slurm ; quota -m; cd thesis; bash squeue_details.sh | grep fstablum ; sbatch -J '$@' -o $TIMESTAMP.out -e $TIMESTAMP.err -D . --constraint='$NODETYPE' --time='$AVAILABLEHOURS:00:00' job.sh $@ ; sleep $SLEEPTIME; tail -n 100 -f $TIMESTAMP.out & tail -n 100 -f $TIMESTAMP.err "
 

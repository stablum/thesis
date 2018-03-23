#!/bin/bash
KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
SSH="ssh -i $KEY fstablum@$HOST"
SLEEPTIME=30
ssh-add $KEY
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
git add -u
git add $1
git diff --cached
echo -n "commit message:"
read COMMIT_MESSAGE
git commit -m "$COMMIT_MESSAGE"
sshfs fstablum@$HOST:/home/fstablum ~/das4mount
git push das4mount master
pushd ~/das4mount/
cd thesis
git stash
popd

# kill the job before 7 AM
AVAILABLEHOURS=$(expr 24 + 6 - $(date +%H))

#NODETYPE="ngpus=1"
#NODETYPE="GTX680"
#NODETYPE="K20"
#NODETYPE="C2050"
NODETYPE="GTX480"
echo "NODETYPE:$NODETYPE"
echo "TIMESTAMP:$TIMESTAMP"
echo "after submitting the job will sleep for $SLEEPTIME seconds before retrieving outputs"
$SSH "module load slurm ; quota -m; cd thesis; squeue | grep fstablum ; sbatch -J '$@' -o $TIMESTAMP.out -e $TIMESTAMP.err -D . --constraint='$NODETYPE' --time='$AVAILABLEHOURS:00:00' job.sh -- $@ ; sleep $SLEEPTIME; tail -f $TIMESTAMP.out & tail -f $TIMESTAMP.err "
 

#!/bin/bash
KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
SSH="ssh -i $KEY fstablum@$HOST"
SLEEPTIME=30
ssh-add $KEY
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
git diff
git add -u
git add $1
echo -n "commit message:"
read COMMIT_MESSAGE
git commit -m "$COMMIT_MESSAGE"
git push das4vu master
NODETYPE="fat,gpu=K20"
#NODETYPE="gpu=C2050"
echo "after submitting the job will sleep for $SLEEPTIME seconds before retrieving outputs"
$SSH "quota -m; cd thesis; git stash; qstat | grep fstablum ; qsub -N '$@' -o $TIMESTAMP.out -e $TIMESTAMP.err -cwd -l $NODETYPE -l h_rt=96:00:00 job.sh -- $@ ; sleep $SLEEPTIME; tail -f $TIMESTAMP.out & tail -f $TIMESTAMP.err "
 

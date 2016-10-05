#!/bin/bash
KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
SSH="ssh -i $KEY fstablum@$HOST"
SLEEPTIME=30
ssh-add $KEY
TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
git add -u
echo -n "commit message:"
read COMMIT_MESSAGE
git commit -m "$COMMIT_MESSAGE"
git push das4vu master
echo "after submitting the job will sleep for $SLEEPTIME seconds before retrieving outputs"
$SSH "cd reimplementations; git stash; qstat | grep fstablum ; qsub -o $TIMESTAMP.out -e $TIMESTAMP.err -cwd -l gpu=C2050 -l h_rt=24:00:00 job.sh -- $@ ; sleep $SLEEPTIME; tail -f $TIMESTAMP.out & tail -f $TIMESTAMP.err "
 

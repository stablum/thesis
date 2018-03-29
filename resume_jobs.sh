#!/bin/bash

function myecho {
    RED='\033[0;31m'
    NC='\033[0m' # No Color
    echo -e "$RED$@$NC"
}

myecho "mount sftp.."
bash mount_sftp.sh

myecho "sync src.."
bash sync_src.sh

myecho "pushd in the das4mount.."
pushd ~/das4mount
cd thesis

SLEEPTIME=10
HARVESTS=$(python3.6m list_harvest_with_state.py --automatic-max-epoch)
TOTAL=$(myecho $HARVESTS | wc -w)
myecho "HARVESTS=$HARVESTS"
myecho "looping on the harvest dirs.. (pwd=$(pwd))"
COUNTER=0
for HARVEST_DIR in $HARVESTS ; do
    let COUNTER=COUNTER+1
    myecho "($COUNTER/$TOTAL) resuming job on $HARVEST_DIR ..."
    bash engage.sh resume_one.sh $HARVEST_DIR &
    myecho "($COUNTER/$TOTAL) sleeping $SLEEPTIME ... "
    sleep $SLEEPTIME
done
popd

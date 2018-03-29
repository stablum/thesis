#!/bin/bash

echo "mount sftp.."
bash mount_sftp.sh

echo "sync src.."
bash sync_src.sh

echo "pushd in the das4mount.."
pushd ~/das4mount
cd thesis

SLEEPTIME=10
RED='\033[0;31m'
NC='\033[0m' # No Color
HARVESTS=$(python3.6m list_harvest_with_state.py --automatic-max-epoch)
echo -e "$RED HARVESTS=$HARVESTS"
echo -e "$RED looping on the harvest dirs.. (pwd=$(pwd)) $NC"
for HARVEST_DIR in $HARVESTS ; do
    echo -e "$RED resuming job on $HARVEST_DIR ... $NC"
    #bash engage.sh resume_one.sh $HARVEST_DIR &
    echo -e "$RED sleeping $SLEEPTIME ... $NC"
    sleep $SLEEPTIME
done
popd

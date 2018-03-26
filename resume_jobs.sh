#!/bin/bash

echo "mount sftp.."
bash mount_sftp.sh

echo "sync src.."
bash sync_src.sh

echo "pushd in the das4mount.."
pushd ~/das4mount
cd thesis

echo "looping on the harvest dirs.."
python3.6m list_harvest_with_state.py --automatic-max-epoch | while read HARVEST_DIR ; do
    echo resuming job on $HARVEST_DIR ...
    bash engage.sh resume_one.sh $HARVEST_DIR &
done
popd

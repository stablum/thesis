#!/bin/bash

bash mount_sftp.sh
bash sync_src.sh

pushd ~/das4mount
cd thesis
python3.6m list_harvest_with_state.py --automatic-max-epoch | while read HARVEST_DIR ; do
    echo resuming job on $HARVEST_DIR ...

done
popd

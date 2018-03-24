#!/bin/bash

bash mount_sftp.sh
bash sync_src.sh

pushd ~/das4mount
cd thesis
python3.6m list_harvest_with_state.py
popd

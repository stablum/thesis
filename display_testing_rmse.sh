#!/bin/bash

bash list_harvest.sh | while read x ; do
    FILENAME=$x/testing_rmse.png
    echo FILENAME:$FILENAME
    eog $FILENAME
done

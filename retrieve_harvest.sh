#!/bin/bash

KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl

bash mount_sftp.sh

bash list_harvest.sh | while read x ; do
    echo "retrieve_harvest: processing directory $x ...."
    echo "copying.."
    rsync --info=progress2 --no-W -vvr --exclude="*.pickl*" ~/das4mount/thesis/$x/ ./$x/
    echo "copying done. making plots.."
    #bash make_png_plots_in_dir.sh $x
    echo "retrieve_harvest: processing directory $x done."
done


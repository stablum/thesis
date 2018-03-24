#!/bin/bash

KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl

bash list_harvest.sh | while read x ; do
    scp -r -v -i $KEY fstablum@$HOST:thesis/$x .
    bash make_png_plots_in_dir.sh $x
done


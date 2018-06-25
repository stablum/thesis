#!/bin/bash

python3 list_harvest_with_state.py | while read x ; do echo -n "$x " ; cat $x/*.log | grep epoch | wc -l ; done

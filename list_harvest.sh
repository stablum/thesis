#!/bin/bash
#cat ~/christos_notes.txt \
cat ./experiments_fixed_testing.txt \
    | grep '//' \
    | sed 's/.*\/\///g' \
    | grep harvest \
    | sed 's/^ *//g' \

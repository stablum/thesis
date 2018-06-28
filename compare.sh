#!/bin/bash

function write_rmses {
    echo $2 > $3
    cat $1/*.log | grep "$2 RMSE" | sed "s/.*$2 RMSE://g" >> $3
}

for i in $( seq "${#@}" ) ; do
    CURR=$(eval echo \$$i)
    write_rmses $CURR training /tmp/tr$i.txt
    write_rmses $CURR testing /tmp/te$i.txt
done
TRAINING_FILES=""
TESTING_FILES=""
for i in $( seq "${#@}" ) ; do
    TRAINING_FILES="$TRAINING_FILES /tmp/tr$i.txt"
    TESTING_FILES="$TESTING_FILES /tmp/te$i.txt"
done
(
paste  $TRAINING_FILES $TESTING_FILES
) | cat -n | less 
    

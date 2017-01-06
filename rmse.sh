#!/bin/bash

DIRNAME="$1"

touch /tmp/training_rmsse
touch /tmp/testing_rmse

echo "finding training..."
cat $DIRNAME/*.log | grep training | sed 's/.*://g' > /tmp/training_rmse
echo "finding testing..."
cat $DIRNAME/*.log | grep testing | sed 's/.*://g' > /tmp/testing_rmse

echo "pasting..."

paste /tmp/training_rmse /tmp/testing_rmse | cat -n

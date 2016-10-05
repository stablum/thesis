#!/bin/bash

bash dat2csv.sh ml-1m/ratings.dat
bash shuffle.sh ml-1m/ratings.dat.csv


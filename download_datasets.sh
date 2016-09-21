#!/bin/bash

wget -N http://files.grouplens.org/datasets/movielens/ml-1m.zip
wget -N http://files.grouplens.org/datasets/movielens/ml-100k.zip

for x in *.zip ; do
    unzip "$x"
    rm -vf "$x"
done

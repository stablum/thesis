#!/bin/bash
cd $1
for x in *.py ; do diff -u $x ../$x ; done

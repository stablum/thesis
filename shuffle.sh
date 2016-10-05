#!/bin/bash

tail -n +2 "$1" > "$1.headerless"
head -n 1 "$1" > "$1.shuf"
shuf "$1.headerless" >> "$1.shuf"
rm "$1.headerless"

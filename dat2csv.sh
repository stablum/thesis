#!/bin/bash

echo "userId,movieId,rating,timestamp" > "$1.csv"
cat "$1" | sed 's/::/,/g' >> "$1.csv"

#!/bin/bash

function pyname {
    BASENAME=$(echo $1 | sed 's/harvest_//g' | sed 's/_.*//g' )
    PYNAME=$BASENAME.py
    echo $PYNAME
}
echo entering directory $1 ..
pushd $1
    python3 $(pyname $1) --resume-state
popd


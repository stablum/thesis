#!/bin/bash

function pyname {
    BASENAME=$(echo $1 | sed 's/harvest_//g' | sed 's/_.*//g' )
    PYNAME=$BASENAME.py
    echo $PYNAME
}
pushd $1
    python3 $(pyname $1) --resume-state
popd


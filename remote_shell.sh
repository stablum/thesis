#!/bin/bash
KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
SSH="ssh -v -i $KEY fstablum@$HOST"
SLEEPTIME=30
ssh-add $KEY
$SSH

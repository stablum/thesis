#!/bin/bash

KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
ssh-add $KEY
sshfs fstablum@$HOST:/home/fstablum ~/das4mount


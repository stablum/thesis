#!/bin/bash

KEY=~/.ssh/das4vu
HOST=fs0.das4.cs.vu.nl
MOUNTPOINT=~/das4mount
ssh-add $KEY
if test $(mount | grep $(basename $MOUNTPOINT) | wc -l) -eq 1 ; then
    echo "unmounting das4 mount.."
    echo "Hello123!" | sudo -S umount $MOUNTPOINT
    echo "unmounting done."
fi
sshfs fstablum@$HOST:/home/fstablum $MOUNTPOINT -o follow_symlinks

if test $(ls $MOUNTPOINT | wc -l ) -eq 0 ; then
    echo "WARNING!! unable to mount on $MOUNTPOINT !!!!!!!"
    exit
fi


#!/bin/bash

git add -u
git add $1
git diff --cached
echo -n "commit message:"
read COMMIT_MESSAGE
git commit -m "$COMMIT_MESSAGE"
bash mount_sftp.sh
git push das4mount master
pushd ~/das4mount/
cd thesis
git stash
popd


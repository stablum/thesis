#!/bin/bash

git add -u
git add $1
git diff --cached
if test $(git diff --cached | wc -l) -ne 0 ; then
    echo -n "commit message:"
    read COMMIT_MESSAGE
    git commit -m "$COMMIT_MESSAGE"
    bash mount_sftp.sh
    git push das4mount master
else
    echo "nothing to send to remote repository"
fi

pushd ~/das4mount/
cd thesis
git stash
popd

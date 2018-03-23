#!/usr/bin/env python3.6m

import os
import glob
import sys
import argparse
import re
import shutil
import time

def main():
    parser = argparse.ArgumentParser(description='list harvest dirs with state dir')
    parser.add_argument('--max-epoch', type=int)
    parser.add_argument('--automatic-max-epoch',action='store_true')
    parser.add_argument('--invert',action='store_true')
    parser.add_argument('--delete-state-dir',action='store_true')

    args = parser.parse_args(sys.argv[1:])
    candidates = glob.glob("harvest_*")
    for c in candidates:
        if not os.path.isdir(c):
            continue
        state_dir = os.path.join(c,"state")
        if not os.path.isdir(state_dir):
            continue
        if args.max_epoch or args.automatic_max_epoch:
            with open(os.path.join(state_dir,"epoch")) as f:
                epoch = int(f.read())
                if args.max_epoch:
                    max_epoch = args.max_epoch
                elif args.automatic_max_epoch:
                    with open(os.path.join(c,"config.py")) as config_f:
                        for line in config_f:
                            match = re.match("n_epochs *= *([0-9]+)",line)
                            if match:
                                max_epoch = int(match.groups()[0])
                threshold = max_epoch - 2
                if args.invert:
                    if epoch <= threshold:
                        continue
                else:
                    if epoch > threshold:
                        continue
        if args.delete_state_dir:
            dir_in_tmp = os.path.join('/tmp','deleted_state_{}'.format(time.time()))
            shutil.move(state_dir,dir_in_tmp)
        print(c)

if __name__=="__main__":
    main()


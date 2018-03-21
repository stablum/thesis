import argparse
import re

import logs_summarizer as ls

harvest_dir_parser = re.compile("([\s\t]*)\/\/[\s\t]*(harvest[^\s\t]*)[\s\t]*")

def process(filename):
    with open(filename,"r") as f:
        for line_nl in f.readlines():
            line = line_nl.rstrip()
            print(line)
            matches = harvest_dir_parser.match(line)
            if matches is not None:
                spaces,d = matches.groups()
                stuff = ls.process_single_harvest(d)
                if stuff is None:
                    print(spaces,"None??")
                else:
                    #print(spaces,"this is a harvest dir:",d)
                    print(
                        spaces+"  "+str(stuff['best_testing_rmse']),
                        "train:"+str(stuff['best_training_rmse'])
                    )
def main():
    parser = argparse.ArgumentParser(description='Logs summarizer.')
    parser.add_argument(
        'filename',
        type=str,
        help='text file containing directory paths'
    )
    args = parser.parse_args()
    process(args.filename)

if __name__=="__main__":
    main()

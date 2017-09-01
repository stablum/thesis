#!/usr/bin/env python3
import sys
import re
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import argparse

quantity_re = re.compile("^\d+:\d+:\d+\s+\d+:\d+:\d+\s+(.*):\s+(.*)$")

parser = argparse.ArgumentParser(description="plot quantities from log file")
parser.add_argument('filename')
parser.add_argument('quantity_name')
parser.add_argument('--save')
args = parser.parse_args()

def usage():
    print("usage: {} filename quantity_name".format(sys.argv[0]))

def collect_quantities(filename,quantity_name):
    handle = open(filename,"r")
    quantities = []
    for row in handle:
        tmp = quantity_re.match(row.strip())
        if tmp is None:
            pass # did no match
        else:
            found_quantity_name = tmp.group(1)
            found_quantity_str = tmp.group(2)
            if found_quantity_name == quantity_name:
                quantity = float(found_quantity_str)
                quantities.append(quantity)
    return np.array(quantities)

def main():
    quantities = collect_quantities(args.filename, args.quantity_name)
    print(quantities)
    print(len(quantities))
    x = np.array(range(len(quantities)))
    plt.plot(x,quantities)
    plt.title(args.quantity_name)
    ax = plt.gca()
    ax.grid(True)
    yloc = matplotlib.ticker.LinearLocator(numticks=20.0) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(yloc)

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()

if __name__ == "__main__":
    main()

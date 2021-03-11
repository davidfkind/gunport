#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
Reclassifies numpy solution files for the specified board dimensions.
This results in solutions being converted into image files.

Command line usage example for 6 rows x 6 columns:
    py -3 reclassify.py BA 6x6
    py -3 reclassify.py GA 8x10
"""

import os
import sys
import argparse
import re                       # Regular expression library
import domplotlib as dpl        # Domino plotting library
import common as cmn            # Common defines and functions

__author__ = 'David Kind'
__date__ = '30-01-2020'
__version__ = '1.3'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)



def main(grid, algo):
    """
    Main reclassification script.
    :param grid: grid dimensions in rows and columns.
    :param algo: algorithm to be reclassified.
    :return: execution time
    """
    rows = int(grid[0])
    cols = int(grid[1])
    # Calculate solution maximum number of holes.
    (max_holes, max_dominoes) = cmn.calc_solution_holes(rows, cols)
    #
    # Instantiate the plotting object.
    #
    dplobj = dpl.CPlot(rows, cols, algo, max_holes, max_dominoes)
    #
    # Scan the results folder solutions and update the class members
    #
    print("Scanning the previous results.")
    dplobj.load_soln_info()
#    print(dplobj)  # debug output
    #
    # Scan the results folder for image files and remove any it finds
    #
    print("Removing old image solution files.")
    for fname in os.listdir(dplobj.path_str):
        if fname.endswith(".png"):
            os.remove(os.path.join(dplobj.path_str, fname))
    #
    # Parse the solution files; classify the solutions and plot them.
    #
    print("Re-classfying the results.")
    dplobj.classify_solns()
    print("Plotting all the results.")
    dplobj.plot_all()
    print("Found {} total solutions and {} fundamental solutions" \
          .format(dplobj.total_all_solns, dplobj.total_fundamental))
    #
    # Update the test information file with the correct number of fundamental solutions.
    #
    fname = os.path.join(dplobj.path_str, cmn.RESULTS_FILE)
    if os.path.exists(fname):
        print("Updating the number of fundamental solutions in {}".format(fname))
        # The file exists so read in the entire contents
        with open(fname, "r") as fin:
            contents = fin.read()
        # Update the number of fundamental solutions
        repl = "Fundamental solutions: {}".format(dplobj.total_fundamental)
        # Had to use regex '(?!...) look behind so that regex it didn't consume
        # the 'Fundamental solutions: ' string.
        contents = re.sub(r"(Fundamental solutions:\s*[0-9]+)", repl, contents, re.IGNORECASE)
        # Write the file back again.
        with open(fname, "w") as fout:
            fout.write(contents)
    # Done.


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--version', action='version', version=SCRIPTINFO)
    PARSER.add_argument('algorithm', nargs=1,
                        help='Algorithm results to be reclassified: BA, GA or MA.')
    PARSER.add_argument('grid', nargs=1,
                        help='Gunport problem grid size, eg 10x8.')
    # Get the arguments dictionary, where arguments are the keys.
    ARGS = vars(PARSER.parse_args())
    # Extract the grid dimensions as a list [x, y]
    GRID = ARGS['grid']
    GRID = GRID[0].lower().split("x")
    # The specified algorithm: BA, GA or MA.
    ALGORITHM = ARGS['algorithm'][0]

    # Start up the application
    main(GRID, ALGORITHM)

# EOF

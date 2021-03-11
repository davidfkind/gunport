#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
This script attempts to find all the solutions to the (8 x 10) board by
using the solutions from lower dimensions. This script is dependent on using
the solutions from the results folder for the BA script.

Command line usage example:
    py -3 8x10solns.py
"""

import os
import sys
import argparse
import numpy as np
import domplotlib as dpl        # Domino plotting library
import findallsolns as fas      # Find all solns from fundamental soln
import common as cmn            # Common defines and functions

__author__ = 'David Kind'
__date__ = '21-02-2020'
__version__ = '1.0'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)

def check(brd):
    """
    Checks the board, 2D numpy arrary to ensure holes are placed correctly.
    :param:  brd - 2D numpy array of a populated board.
    :return: boolean valid flag, True = Valid shape placements
    """
    valid = True
    # Check if holes are adjacent to each other.
    (rows, cols) = brd.shape
    for row in range(rows):
        for col in range(cols):
            # Check to see if there is another hole either below or the right.
            if brd[row][col] == cmn.CELL_SPACE:
                # Check shape to the right
                if (col + 1) != cols and brd[row][col + 1] == cmn.CELL_SPACE:
                    valid = False
                # Check shape below
                if (row + 1) != rows and brd[row + 1][col] == cmn.CELL_SPACE:
                    valid = False
        if not valid:
            break
    return valid

def get_solns(src1, src2, src3):
    """
    Loops through each slice to generate complete boards; tests the resultant
    boards to make sure they are valid before adding them to the solution list.
    :param: src1: list of solutions for slice1
    :param: src2: list of solutions for slice2
    :param: src3: list of solutions for slice3
    :return: list of valid board solutions
    """
    solns = []
    for slice1 in src1:
        for slice2 in src2:
            for slice3 in src3:
                # Now add the slices to the (8 x 10) board
                board = np.concatenate([slice1, slice2, slice3])
                # Is the board valid?
                # Note: don't need to calculate the maximum number of holes
                # and minimum number of dominoes, because these are already
                # optimal and correct!
                if check(board):
                    # Add the solution board to the list
                    solns.append(board)
    return solns


def main():
    """
    Main function entry point to calculating all the solutions to the (8 x 10)
    board, by using the solutions from lower dimensions. These being 2x slices
    of (3 x 8) solutions, plus 1x slice of (4 x 8) solutions.
    :param:  n/a
    :return: n/a
    """
    #
    # Check that the solution folders are present, no point continuing if
    # they are not there.
    #
    ok_to_proceed = True
    soln_folders = ["3x8__8_holes-8_dominoes", "4x8__10_holes-11_dominoes"]
    for folder in soln_folders:
        if not os.path.exists(os.path.join(cmn.BA_FOLDER, folder)):
            print("Error: {} doesn't exist.".format(folder))
            ok_to_proceed = False
        else:
            print("Found solution folder '{}'".format(folder))
    if not ok_to_proceed:
        print("Missing solution folder(s) so quitting.")
        sys.exit(-1)
    #
    # Load all the solutions into memory for speed and ease of processing.
    #
    solns = {"3x8": [], "4x8": []}
    for folder in soln_folders:
        path_str = os.path.join(cmn.BA_FOLDER, folder)
        for fname in os.listdir(path_str):
            # Identify a solution file and load it.
            if fname.endswith(".npy"):
                board = np.load(os.path.join(path_str, fname))
                if folder == "3x8__8_holes-8_dominoes":
                    solns["3x8"].append(board)
                else:
                    solns["4x8"].append(board)

    # Need 3x nested for loops representing each of the 3 slices.
    # As the seed board solutions are different we need to execute the nested
    # loop three times to get full coverage, as follows:
    all_solns = []
    # 1) slice1: 3x8, slice2: 3x8, slice3: 4x8
    all_solns.extend(get_solns(solns["3x8"], solns["3x8"], solns["4x8"]))
    # 2) slice1: 3x8, slice2: 4x8, slice3: 3x8
    all_solns.extend(get_solns(solns["3x8"], solns["4x8"], solns["3x8"]))
    # 3) slice1: 4x8, slice2: 3x8, slice3: 3x8
    all_solns.extend(get_solns(solns["4x8"], solns["3x8"], solns["3x8"]))
    
    # Plot the domino board solutions; saving to a each to a file.
    # Note we need to pass in a numpy 2D array!
    max_holes, min_dominoes = cmn.calc_solution_holes(8, 10)
    dplobj = dpl.CPlot(8, 10, "pattern", max_holes, min_dominoes)
    for result in all_solns:
        dplobj.write_soln_to_file(result)
    dplobj.classify_solns()
    dplobj.plot_all()



if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--version', action='version', version=SCRIPTINFO)

    # Start up the application
    main()

# EOF

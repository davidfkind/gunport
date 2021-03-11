#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#

"""
Common code and defines library module for The Gun Port Problem solvers.
The Gun Port Problem as first described by
Sands, B. (1971), The Gunport Problem, Mathematics Magazine, Vol.44, pp.193-196
"""

import os
import sys
import platform
import numpy as np
import findallsolns as fas      # Find all solns from fundamental soln (debug)

__author__ = 'David Kind'
__date__ = '25-02-2020'
__version__ = '2.1'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Grid domino type definitions
#
CELL_SPACE = 0                  # Empty square
CELL_HDOMINO = 1                # Horizontal domino
CELL_VDOMINO = 2                # Vertical domino
CELL_UNASSIGNED = 3             # Cell location is unassigned

# Common results folder definition
RESULTS_FOLDER = "results"
RESULTS_FILE = "test.info.txt"
BA_FOLDER = os.path.join(RESULTS_FOLDER, "BA")
GA_FOLDER = os.path.join(RESULTS_FOLDER, "GA")
MA_FOLDER = os.path.join(RESULTS_FOLDER, "MA")
MA_PARTIAL = os.path.join(RESULTS_FOLDER, "MA.partial")
RESULTS_FOLDERS = (BA_FOLDER, GA_FOLDER, MA_FOLDER, MA_PARTIAL)
PERMUTATIONS_FOLDER = os.path.join(RESULTS_FOLDER, "PERMUTATIONS")
PERMUTATIONS_FILE = os.path.join(RESULTS_FOLDER, "permutations.txt")


def calc_solution_holes(x_coord, y_coord):
    """
    Ref: Knotted Doughnuts and Other Mathematical Entertainments (Book)
         Martin Gardner 1986
    In his book the author summarises the maximum number of holes as dependent
    on the grid x and y sizes.
    1) If x or y are divisible by 3 then
            holes = (x * y) / 3
    2) If x or y both equal (3 * k + 1) or both equal (3 * k + 2) then
            holes = (x * y - 4) / 3
    3) If x or y are equal to (3 * k + 1) and (3 * k + 2) then
            holes = (x * y - 2) / 3
    This function calculates the maximum number of holes given the above and
    the passed in parameter corresponding to the x and y grid dimensions.
    :param x_coord: maximum number of horizontal cells in the grid.
    :param y_coord: maximum number of vertical cells in the grid.
    :return: returns the maximum number of holes and dominoes.
    """
    # 1) Are x_coord or y_coord divisible by 3
    if not x_coord % 3 or not y_coord % 3:
        holes = int((x_coord * y_coord) / 3)
    else:
        # Note: to get here we know that neither x_coord or y_coord are
        # divisible by 3.
        x_const = x_coord - (int(x_coord / 3) * 3)
        y_const = y_coord - (int(y_coord / 3) * 3)
        # 2) If x_coord or y_coord both have equal constants
        if x_const == y_const:
            holes = int((x_coord * y_coord - 4) / 3)
        # 3) If x_coord or y_coord both have unequal constants
        else:
            holes = int((x_coord * y_coord - 2) / 3)
    # Now calculate the maximum number of dominoes
    dominoes = int(((x_coord * y_coord) - holes) / 2)
    return holes, dominoes


def get_version_str(algorithm_str):
    """
    Common method to retrieve the version string of the running script.
    :param: algorithm_str - name of algorithm to log with the version info.
    :return: returns version string.
    """
    # Get the Python version information
    verstr = "The following results were generated using {}.\n" \
             .format(algorithm_str)
    verstr += " - Python version: {}.{}.{}\n".format(sys.version_info[0],
                                                     sys.version_info[1],
                                                     sys.version_info[2])
    verstr += " - Test PC: {} {}\n" \
              .format(platform.system(), platform.release())
    return verstr

def convert2board(chrom, rows, cols):
    """
    Converts the chromosome represented in a list into a 2D numpy array.
    :param rows: number of rows associated with the board.
    :param cols: number of columns associated with the board.
    :param chrom: chromosome to be converted.
    :return: 2D numpy array.
    """
    # Initialise the variables to be used
    idx = int(0)    # Chromosome index
    board = np.zeros((rows, cols), 'uint8')
    board.fill(CELL_UNASSIGNED)
    # Now loop through the board adding the shapes and checking validity.
    # Start at top left corner, processing each row in turn.
    for row in range(rows):
        for col in range(cols):
            # Retrieve the next shape
            shape = chrom[idx]
            # Skip the cell if it is already occupied.
            if board[row][col] != CELL_UNASSIGNED:
                continue
            # Have we run out of shapes...
            if shape == CELL_UNASSIGNED:
                idx = idx + 1
                if idx >= len(chrom):
                    return board
                continue
            # Attempt to place the shape on the board.
            if shape == CELL_SPACE:
                # Place the hole if valid.
                if not ((col > 0 and board[row][col - 1] == CELL_SPACE) or
                        (row > 0 and board[row - 1][col] == CELL_SPACE)):
                    board[row][col] = CELL_SPACE
            elif shape == CELL_HDOMINO:
                # Are we ok to have a horizontal domino?
                if col < cols - 1 and board[row][col + 1] == CELL_UNASSIGNED:
                    board[row][col] = CELL_HDOMINO
                    board[row][col + 1] = CELL_HDOMINO
            else:
                # shape == CELL_VDOMINO:
                # Are we ok to have a vertical domino?
                if row < rows - 1:
                    board[row][col] = CELL_VDOMINO
                    board[row + 1][col] = CELL_VDOMINO
            # Move on to the next shape
            idx = idx + 1
            if idx >= len(chrom):
                return board
    return board

def convert2chrom(board, csize=None):
    """
    Converts the board represented as a 2D numpy array into a list.
    :param board: populated board to be converted into a chromosome.
    :param csize: chromosome size; else just fits to the chromosome.
    :return: chromosome list.
    """
    # Take a copy of the 2D array.
    brd = np.copy(board)
    (rows, cols) = np.shape(board)
    # Now loop through the board extracting the shapes and adding to the
    # chromosome. Clear the pieces as we scan through the board.
    # Start at top left corner, processing each row in turn.
    chrom = []
    for row in range(rows):
        for col in range(cols):
            # Skip the cell if it is already unassigned, already processed.
            if brd[row][col] == CELL_UNASSIGNED:
                continue
            # Retrieve the next shape
            shape = brd[row][col]
            chrom.append(shape)
            # Clear the shape, it's been processed
            brd[row][col] = CELL_UNASSIGNED
            if shape == CELL_HDOMINO:
                brd[row][col + 1] = CELL_UNASSIGNED
            elif shape == CELL_VDOMINO:
                brd[row + 1][col] = CELL_UNASSIGNED
    # Adjust if the chromosome size has been specified.
    if csize:
        adjusted = []
        for idx in range(csize):
            if idx < len(chrom):
                adjusted.append(chrom[idx])
            else:
                adjusted.append(CELL_UNASSIGNED)
        chrom = adjusted
    return chrom


if __name__ == '__main__':
    # Test the common functions:
    # > calc_solution_holes()
    # > convert2board()
    # > convert2chrom()

    # Create a (3 x 3) board with a solution
    # This has 1 fundamental solution and 4 solutions in total.
    # It results in 3 holes and 3 dominoes.
    ROWS = 3
    COLS = 3
    # Retrieve the expected number of holes and dominoes
    EXPECTED_HOLES = 3
    EXPECTED_DOMINOES = 3
    (HOLES, DOMINOES) = calc_solution_holes(ROWS, COLS)
    if HOLES != EXPECTED_HOLES or DOMINOES != EXPECTED_DOMINOES:
        print("calc_solution_holes() failed.")
        print("It returned {} holes, expected {}"
              .format(HOLES, EXPECTED_HOLES))
        print("It returned {} dominoes, expected {}"
              .format(DOMINOES, EXPECTED_DOMINOES))
    else:
        print("calc_solution_holes() correct.")

    # New solution
    # Fill the board up with holes
    #
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = CELL_HDOMINO
    BOARD[0, 2] = CELL_HDOMINO
    BOARD[1, 0] = CELL_HDOMINO
    BOARD[1, 1] = CELL_HDOMINO
    BOARD[2, 1] = CELL_HDOMINO
    BOARD[2, 2] = CELL_HDOMINO

    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        # Can now test conversion to chromosome and board.
        print("\n" + "-" * 40)
        print("board = \n{}".format(solution))
        CHROM = convert2chrom(solution)
        print("resultant chromosome = {}".format(CHROM))
        BOARD = convert2board(CHROM, ROWS, COLS)
        print("resultant board = \n{}".format(BOARD))
        if np.array_equal(solution, BOARD):
            print("Conversion successful")
        else:
            print("CONVERSION FAILED!!!")
    # Check that covert2chrom can fill undersized chromosomes
    TEST_CHROM = [CELL_UNASSIGNED] * (ROWS * COLS)
    for idx, val in enumerate(CHROM):
        TEST_CHROM[idx] = val
    print("\nChromosome padding test.")
    print("Before: {}".format(CHROM))
    CHROM = convert2chrom(solution, (ROWS * COLS))
    print("After: {}".format(CHROM))
    if TEST_CHROM == CHROM:
        print("convert2chrom() successfully executed.")
    else:
        print("convert2chrom() fill failed!!!")
    print("\n")

    # Testing Pearce equations against Gyarfas equations.
    for n in range(16):
        holes, dominoes = calc_solution_holes(n, n)
        gdominoes = int(((n * n) + 2) / 3)
        gholes = ((n * n) - (2 * gdominoes))
        print("{}".format(n) + "-" * 50)
        print("Pearce holes  = {}, dominoes = {}".format(holes, dominoes))
        print("Gyarfas holes = {}, dominoes = {}".format(gholes, gdominoes))
        if gdominoes != dominoes:
            break

# EOF

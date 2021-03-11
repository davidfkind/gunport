#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
Find all solutions script.
Written for use with the Gunport Problem solving scripts.
"""

import numpy as np
import common as cmn            # Common defines and functions

__author__ = 'David Kind'
__date__ = '30-01-2020'
__version__ = '1.6'
__copyright__ = 'Copyright(c) 2019 De Montfort University. All rights reserved.'


def findall(board):
    """
    Takes the solution board as an input, this is a numpy ndarray and then
    performs rotations and flips to extract all the possible solutions.
    Parameters:
        board:       encoded numpy ndarray of dominoes fitted to the board.
    Returns:
        A list of all the solutions found; these are the numpy ndarrays'.
    """
    # Keep track of all the solutions we have found
    all_solns = list()
    # Add the initial solution and treat this as the fundamental solution.
    all_solns.append(board)
    # Rotate the board to find new solutions
    all_solns = domino_rotation(all_solns, board)
    # Re-run the rotations but with a flipped/mirrored board
    fboard = np.fliplr(board)
    # Add the new solution if it does not already exist in the solutions list.
    if True not in [np.array_equal(fboard, soln) for soln in all_solns]:
        all_solns.append(fboard)
    # Rotate the board to find new solutions
    all_solns = domino_rotation(all_solns, fboard)
    # Check for a square, 2x dominoes together, as there could be several and
    # then check rotations. Get a list of boards with any squares.
    squares = domino_squares(board)
    for square in squares:
        if True not in [np.array_equal(square, soln) for soln in all_solns]:
            all_solns.append(square)
        else:
            # This solution already exists, try the next one.
            continue
        # Rotate the board to find new solutions
        all_solns = domino_rotation(all_solns, square)
        # Re-run the rotations but with a flipped/mirrored board
        fboard = np.fliplr(square)
        # Add the new solution if it does not already exist in the solutions list.
        if True not in [np.array_equal(fboard, soln) for soln in all_solns]:
            all_solns.append(fboard)
        else:
            # This solution already exists, try the next one.
            continue
        # Rotate the board to find new solutions
        all_solns = domino_rotation(all_solns, fboard)
    return all_solns


def domino_correction(board):
    """
    Simply parses a numpy ndarray and converts 1s' to 2s' and 2s' to 1s'
    returning the result back to the calling function.
    Parameters:
        board:       encoded numpy ndarray of dominoes fitted to the board
    Returns:
        The updated board array.
    """
    # Determine the size/shape of the board array parameter
    (ysize, xsize) = board.shape
    # Parse each board location in turn and convert if necessary
    result = np.zeros((ysize, xsize), 'uint8')
    for x in range(xsize):
        for y in range(ysize):
            if board[y, x] == cmn.CELL_HDOMINO:
                result[y, x] = cmn.CELL_VDOMINO
            elif board[y, x] == cmn.CELL_VDOMINO:
                result[y, x] = cmn.CELL_HDOMINO
    return result


def domino_rotation(asolns, brd):
    """
    Rotate the new solution and add the result to the list of all solutions
    if it unique.
    In order to find all the solutions the fundamental solution will be
    rotated by 90 degrees 3 times. The fundamental solution will be flipped
    and then rotated by 90 degrees 3 times.
    Note: adjusted solutions may have to have the domino orientation
    updated, for example a rotation by 90 degrees means that vertical
    dominoes will have to be changed to horizontal dominoes and horizontal
    dominoes will have to be changed to vertical dominoes. This maintains
    the resultant output plot colour coding.
    :param asolns: list of numpy arrays, all solutions found so far.
    :param brd:    2D numpy array of the board to be rotated.
    :return:       list of numpy arrays, all solutions.
    """
    # Add the new solution if it does not already exist in the solutions list.
    nsoln = domino_correction(np.rot90(brd, 1))
    if True not in [np.array_equal(nsoln, soln) for soln in asolns]:
        asolns.append(nsoln)
    nsoln = np.rot90(brd, 2)
    # Add the new solution if it does not already exist in the solutions list.
    if True not in [np.array_equal(nsoln, soln) for soln in asolns]:
        asolns.append(nsoln)
    nsoln = domino_correction(np.rot90(brd, 3))
    # Add the new solution if it does not already exist in the solutions list.
    if True not in [np.array_equal(nsoln, soln) for soln in asolns]:
        asolns.append(nsoln)
    return asolns

def domino_squares(brd):
    """
    Checks the board for domino squares and returns a list of all the new
    combinations of boards with the squares swapped. These new solutions will
    have to be rotated and checked to see if they in turn provide new solutions.
    :param brd:    2D numpy array of the board to be rotated.
    :return:       list boards with modified squares.
    """
    # Create a simple copy of the board to make it easier to identify squares.
    # Holes are cleared as are the 2nd square of the current domino.
    sbrd = np.copy(brd)
    (rows, cols) = np.shape(sbrd)
    # Now loop through the board clearing all unnecessary locations.
    for row in range(rows):
        for col in range(cols):
            # Retrieve the next shape
            shape = sbrd[row][col]
            # Skip the cell if it is already empty.
            if shape == cmn.CELL_UNASSIGNED:
                continue
            if shape == cmn.CELL_VDOMINO:
                sbrd[row + 1][col] = cmn.CELL_UNASSIGNED
            elif shape == cmn.CELL_HDOMINO:
                sbrd[row][col + 1] = cmn.CELL_UNASSIGNED
            else:
                # Clear the hole, it's been processed
                sbrd[row][col] = cmn.CELL_UNASSIGNED
#    print(sbrd)  # debug
    # Now loop through and find any squares
    squares = []
    for row in range(rows):
        for col in range(cols):
            shape = sbrd[row][col]
            if shape == cmn.CELL_HDOMINO and (row + 1) < rows and \
               sbrd[row + 1][col] == cmn.CELL_HDOMINO:
                # Found 2x horizontal dominoes, convert to 2 vertical dominoes.
                nbrd = np.copy(brd)
                nbrd[row][col] = cmn.CELL_VDOMINO
                nbrd[row][col + 1] = cmn.CELL_VDOMINO
                nbrd[row + 1][col] = cmn.CELL_VDOMINO
                nbrd[row + 1][col + 1] = cmn.CELL_VDOMINO
                squares.append(nbrd)
            elif shape == cmn.CELL_VDOMINO and (col + 1) < cols and \
                sbrd[row][col + 1] == cmn.CELL_VDOMINO:
                # Found 2x vertical dominoes
                nbrd = np.copy(brd)
                nbrd[row][col] = cmn.CELL_HDOMINO
                nbrd[row][col + 1] = cmn.CELL_HDOMINO
                nbrd[row + 1][col] = cmn.CELL_HDOMINO
                nbrd[row + 1][col + 1] = cmn.CELL_HDOMINO
                squares.append(nbrd)
    # It is a current limitation that the code is unable to cater for complex
    # combinations of groups of dominoes together. ie. 3 vertical dominoes
    # together would result in alternating blocks of horizontal dominoes.
    # Ideally we would want to create a list of combinations of multiple
    # squares, when available.
    return squares


def display(solns):
    """
    Displays all the solutions in the array.
    :param solns: numpy array of solutions
    :return: n/a
    """
    print(solns)
    for idx, board in enumerate(solns):
        print("{} ---------------------------".format(idx))
        print("{}".format(board))


if __name__ == '__main__':
    # Note: 0=space/hole, 1=horizontal domino, 2=vertical domino
    # Add a fundamental solution for 3x3 board
    TESTGRID = np.zeros((3, 3), 'uint8')
    TESTGRID[0, 1] = cmn.CELL_VDOMINO
    TESTGRID[1, 0] = cmn.CELL_VDOMINO
    TESTGRID[1, 1] = cmn.CELL_VDOMINO
    TESTGRID[1, 2] = cmn.CELL_VDOMINO
    TESTGRID[2, 0] = cmn.CELL_VDOMINO
    TESTGRID[2, 2] = cmn.CELL_VDOMINO
    display(findall(TESTGRID))
    print("+" * 80)
    # Add a fundamental solution for 4x3 board
    TESTGRID = np.zeros((4, 3), 'uint8')
    TESTGRID[0, 1] = cmn.CELL_VDOMINO
    TESTGRID[1, 0] = cmn.CELL_VDOMINO
    TESTGRID[1, 1] = cmn.CELL_VDOMINO
    TESTGRID[1, 2] = cmn.CELL_VDOMINO
    TESTGRID[2, 0] = cmn.CELL_VDOMINO
    TESTGRID[2, 2] = cmn.CELL_VDOMINO
    TESTGRID[3, 1] = cmn.CELL_HDOMINO
    TESTGRID[3, 2] = cmn.CELL_HDOMINO
    display(findall(TESTGRID))
    print("+" * 80)

    # Add a fundamental solution for 5x5 board [2]-[0] 7 holes, 9 dominoes.
    # Ensure each square is replaced with either horizontal or vertical
    # dominoes. This solution is unusual as it has a square composed of two
    # vertical dominoes. Observation and logic tells us that the two
    # vertical dominoes can be replaced with two horizontal dominoes.
    TESTGRID = np.zeros((5, 5), 'uint8')
    # Board row #1
    TESTGRID[0, 1] = cmn.CELL_HDOMINO
    TESTGRID[0, 2] = cmn.CELL_HDOMINO
    TESTGRID[0, 4] = cmn.CELL_VDOMINO
    # Board row #2
    TESTGRID[1, 0] = cmn.CELL_HDOMINO
    TESTGRID[1, 1] = cmn.CELL_HDOMINO
    TESTGRID[1, 3] = cmn.CELL_VDOMINO
    TESTGRID[1, 4] = cmn.CELL_VDOMINO
    # Board row #3
    TESTGRID[2, 1] = cmn.CELL_VDOMINO
    TESTGRID[2, 2] = cmn.CELL_VDOMINO
    TESTGRID[2, 3] = cmn.CELL_VDOMINO
    # Board row #4
    TESTGRID[3, 0] = cmn.CELL_VDOMINO
    TESTGRID[3, 1] = cmn.CELL_VDOMINO
    TESTGRID[3, 2] = cmn.CELL_VDOMINO
    TESTGRID[3, 4] = cmn.CELL_VDOMINO
    # Board row #5
    TESTGRID[4, 0] = cmn.CELL_VDOMINO
    TESTGRID[4, 2] = cmn.CELL_HDOMINO
    TESTGRID[4, 3] = cmn.CELL_HDOMINO
    TESTGRID[4, 4] = cmn.CELL_VDOMINO
    display(findall(TESTGRID))
    print("+" * 80)

# EOF

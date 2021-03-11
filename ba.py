#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
Backtracking Algorithm to solve The Gun Port Problem as first described by
Sands, B. (1971), The Gunport Problem, Mathematics Magazine, Vol.44, pp.193-196

This is an optimisation problem to maximise the holes and minimise the number
number of dominoes in a grid of variable size. Holes are not allowed to touch
each other side on.

Command line usage example for 6 rows x 6 columns:
    py -3 ba.py 6x6
    py -3 ba.py 8x10 --timer
    py -3 ba.py 8x10 --permutations
"""

import os
import sys
import argparse
import time                     # Used to time script execution.
import multiprocessing as mp
from multiprocessing import Process, Value, Lock
import numpy as np
import domplotlib as dpl        # Domino plotting library
import common as cmn            # Common defines and functions

__author__ = 'David Kind'
__date__ = '16-02-2020'
__version__ = '2.8'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)

# Keep track of the total number of Permutations
# Ref: https://dzone.com/articles/shared-counter-python%E2%80%99s
class Counter():
    """
    Counter Object used for counting the number of permutations when using
    multiple processors.
    """
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        """
        Increment the value by one, safely.
        :return: n/a
        """
        with self.lock:
            self.val.value += 1

    def value(self):
        """
        Read the value, safely.
        :return: integer counter value.
        """
        with self.lock:
            return self.val.value


class CBacktracking():
    """
    Backtracking Algorithm Object.
    """
    rows = int(0)
    cols = int(0)
    chrom_start = int(0)
    maxholes = int(0)
    maxdominoes = int(0)
    board = []

    def __init__(self, rows, cols, dplobj, counter, pflag):
        # Initialise the object members
        self.rows = rows
        self.cols = cols
        self.dplobj = dplobj
        self.chrom_start = 0    # Chromosome starting index
        self.permutations = counter
        self.pflag = pflag      # Permutations calculation flag
        #
        # Calculate solution maximum number of holes.
        #
        (self.maxholes, self.maxdominoes) = cmn.calc_solution_holes(rows, cols)
        # OK taking a different approach so that we use a list of shapes, as per
        # the chromosome and then run all our tests on this list. This has the
        # benefit of not requiring us to manage the shape of the dominoes on a
        # grid, where we'd have to check for overlap and manage that. We also
        # have many of the functions that work on a list already developed and
        # tested.
        #
        # Create the chromosome and fill it with unassigned values.
        # Note: that this is maximum possible list size and is the equivalent of
        # filling the board with spaces only.
        self.chromosome = [cmn.CELL_UNASSIGNED for _ in range(rows * cols)]
        # Initialise the board size and fill with zeros
        self.board = np.zeros((self.rows, self.cols), 'uint8')

    def run(self):
        """
        Execute the backtracking algorithm.
        """
        self.solve(self.chromosome, (self.rows * self.cols))

    def runp(self, chrom_init):
        """
        Execute the backtracking algorithm on a separate processor
        :param chrom_init: chromosome initialisation list.
        """
        # Seed the BA chromosome with initial board Polyominoes.
        self.chrom_start = len(chrom_init)
        for idx in range(self.chrom_start):
            self.chromosome[idx] = chrom_init[idx]
        # Now solve the board using this configuration.
        self.solve(self.chromosome, (self.rows * self.cols))

    def solve(self, chrom, unassigned):
        """
        Recursive backtracking function.
        Ref: CELL_SPACE=0, CELL_HDOMINO=1, CELL_VDOMINO=2, CELL_UNASSIGNED=3
        :param chrom: chromosome implemented as per GA.
        :param unassigned: number of unassigned cells remaining on the board.
        """
        if unassigned <= 0:
            (valid, unassigned) = self.isvalid(chrom)
            if self.pflag:
                if valid:
                    print(".", end=" ")
                    self.permutations.increment()
                return False
            # Not calculating the permutations
            if valid and self.optimal_solution(chrom):
                # Found a solution; write it to a file; process later.
                self.dplobj.write_soln_to_file(self.board)
            return False
        # Process each cell in the chromosome
        for idx in range(self.chrom_start, len(chrom)):
            cell = chrom[idx]
            # Find the next empty Cell
            if cell != cmn.CELL_UNASSIGNED:
                # Cell is occupied, skip it.
                continue
            # Attempt to place each shape in turn into this location
            # Order being: CELL_SPACE, CELL_HDOMINO, CELL_VDOMINO
            #            shape = cmn.CELL_SPACE
            for shape in range(cmn.CELL_UNASSIGNED):
                chrom[idx] = shape
                (valid, unassigned) = self.isvalid(chrom)
#                print("shape[{}]={}, valid={}, unassigned={}" \
#                     .format(idx, shape, valid, unassigned))
#                print("{}\n----------------------".format(self.board))
                if valid and self.solve(chrom, unassigned):             # Recurrent function
#                    print("Success[2]")
                    return True
                # Backtrack
                chrom[idx] = cmn.CELL_UNASSIGNED
            # No point proceeding if have cycled through all the options.
            if shape == cmn.CELL_VDOMINO:
#                print("Failure[2]")
                return False
#        print("Failure[3]")
        return False

    def optimal_solution(self, chrom):
        """
        Verifies the solution with the maximum number of holes and dominoes
        possible for the board dimensions.
        :param chrom: chromosome to be validated by placing shapes on a board.
        :return boolean: flag to show whether shapes are placed legally.
        """
        # Initialise the variables to be used
        holes = int(0)
        dominoes = int(0)
        # Cycle through chromosome; calculate the number of holes and dominoes
        for entry in chrom:
            if entry == cmn.CELL_UNASSIGNED:
                continue
            if entry == cmn.CELL_SPACE:
                holes = holes + 1
            else:
                dominoes = dominoes + 1
        # Verify if we have an optimal solution
        return bool(holes == self.maxholes and dominoes == self.maxdominoes)

    def isvalid(self, chrom):
        """
        Verifies the validity of the chromosome by laying out the dominoes on
        the board and checking for rule compliance: no two holes may be adjacent
        to each other; no overlapping dominoes; no dominoes exceeding the board
        space.
        :param chrom: chromosome to be validated by placing shapes on a board.
        :return boolean: flag to show whether shapes are placed legally.
        :return integer: value indicating the number of unassigned cells
                         remaining.
        """
        # Initialise the variables to be used
        idx = int(0)    # Chromosome index
        unassigned_cells = int(0)
        self.board.fill(cmn.CELL_UNASSIGNED)
        # Now loop through the board adding the shapes and checking validity.
        # Start at top left corner, processing each row in turn.
        for row in range(self.rows):
            for col in range(self.cols):
                # Retrieve the next shape
                shape = chrom[idx]
                # Skip the cell if it is already occupied.
                if self.board[row][col] != cmn.CELL_UNASSIGNED:
                    continue
                # Have we run out of shapes...
                if shape == cmn.CELL_UNASSIGNED:
                    unassigned_cells = unassigned_cells + 1
                    continue
                # Attempt to place the shape on the board.
                if shape == cmn.CELL_SPACE:
                    if self.pflag:
                        # Place the hole; no valid check required.
                        self.board[row][col] = cmn.CELL_SPACE
                    else: # Not calculating the permutations
                        # Place the hole if valid.
                        if not ((col > 0 and self.board[row][col - 1] == cmn.CELL_SPACE) or
                                (row > 0 and self.board[row - 1][col] == cmn.CELL_SPACE)):
                            self.board[row][col] = cmn.CELL_SPACE
                        else:
                            # Can't place the shape
                            unassigned_cells = unassigned_cells + 1
                            return False, unassigned_cells
                elif shape == cmn.CELL_HDOMINO:
                    # Are we ok to have a horizontal domino?
                    if col < self.cols - 1 and self.board[row][col + 1] == cmn.CELL_UNASSIGNED:
                        self.board[row][col] = cmn.CELL_HDOMINO
                        self.board[row][col + 1] = cmn.CELL_HDOMINO
                    else:
                        # Can't place the shape
                        unassigned_cells = unassigned_cells + 1
                        return False, unassigned_cells
                else:
                    # shape == cmn.CELL_VDOMINO:
                    # Are we ok to have a vertical domino?
                    if row < self.rows - 1:
                        self.board[row][col] = cmn.CELL_VDOMINO
                        self.board[row + 1][col] = cmn.CELL_VDOMINO
                    else:
                        # Can't place the shape
                        unassigned_cells = unassigned_cells + 1
                        return False, unassigned_cells
                # Move on to the next shape
                idx = idx + 1
        return True, unassigned_cells

    def __str__(self):
        """
        Object string handling function.
        """
        outstr = "Running {} with grid ({} rows x {} cols):\n" \
                 .format(SCRIPTNAME, self.rows, self.cols)
        outstr += "Max Holes={}, Max Dominoes={}.\n\n" \
                  .format(self.maxholes, self.maxdominoes)
        return outstr

def ba_proc(rows, cols, chrom_seq, pnum, counter, permutations):
    """
    BA Algorithm function for use in a multiprocessor process.
	:param rows:         number of rows on the board.
	:param cols:         number of columns on the board.
	:param chrom_seq:    pre-initialised chromosome.
	:param pnum:         number of processors.
	:param permutations: permutations calculation flag.
	:param counter:      reference to counter object.
    :return: n/a
    """
    #
    # Plot the domino grid solutions; saving to a each to a file.
    #
    print("Saving resultant board plots.")
    # Retrieve the expected number of holes and dominoes
    (holes, dominoes) = cmn.calc_solution_holes(rows, cols)
    dplobj = dpl.CPlot(rows, cols, "BA", holes, dominoes)
    dplobj.set_proc_num(pnum)
    #
    # Create the backtracking algorithm object with the specified board size.
    #
    baobj = CBacktracking(rows, cols, dplobj, counter, permutations)
    # Display the backtracking algorithm's properties
    if pnum == 0:
        print(baobj)    # Only print it the once.
    # Now ready to solve the problem using the chromosome.
    print("Executing the backtracking algorithm on processor[{}].".format(pnum))
    baobj.runp(chrom_seq)

def main(grid, calc_permutations, timed_execution):
    """
    Main Bracktracking Algorithm implementation.
    :param grid: grid dimensions in rows and columns.
    :param timed_execution: flag to time algorithm execution.
    :return: execution time
    """
    start = time.time()        # Used to time script execution.
    rows = int(grid[0])
    cols = int(grid[1])

    # Set the number of parallel processes we want to run.
    # This is dependent on the number of processors available and the
    # initialisation values available and the board dimensions.
    num_processors = mp.cpu_count()
    print("Number of processors available: {}".format(num_processors))
    # Create the chromosome initialisation list.
    if calc_permutations and num_processors >= 9 and rows > 4 and cols > 4:
        num_processors = 9  # Only use 9 processors
        # Note: when calculating the permutations we need ALL combinations!
        chrom_init_seq = [[cmn.CELL_SPACE, cmn.CELL_HDOMINO],
                          [cmn.CELL_SPACE, cmn.CELL_VDOMINO],
                          [cmn.CELL_SPACE, cmn.CELL_SPACE],
                          [cmn.CELL_HDOMINO, cmn.CELL_SPACE],
                          [cmn.CELL_HDOMINO, cmn.CELL_HDOMINO],
                          [cmn.CELL_HDOMINO, cmn.CELL_VDOMINO],
                          [cmn.CELL_VDOMINO, cmn.CELL_SPACE],
                          [cmn.CELL_VDOMINO, cmn.CELL_HDOMINO],
                          [cmn.CELL_VDOMINO, cmn.CELL_VDOMINO]]
    if not calc_permutations and num_processors >= 8 and rows > 4 and cols > 4:
        num_processors = 8  # Only use 8 processors
        chrom_init_seq = [[cmn.CELL_SPACE, cmn.CELL_HDOMINO],
                          [cmn.CELL_SPACE, cmn.CELL_VDOMINO],
                          [cmn.CELL_HDOMINO, cmn.CELL_SPACE],
                          [cmn.CELL_HDOMINO, cmn.CELL_HDOMINO],
                          [cmn.CELL_HDOMINO, cmn.CELL_VDOMINO],
                          [cmn.CELL_VDOMINO, cmn.CELL_SPACE],
                          [cmn.CELL_VDOMINO, cmn.CELL_HDOMINO],
                          [cmn.CELL_VDOMINO, cmn.CELL_VDOMINO]]
    elif not calc_permutations and num_processors >= 6 and rows >= 4 and cols >= 4:
        num_processors = 6  # Only use 6 processors
        chrom_init_seq = [[cmn.CELL_SPACE, cmn.CELL_HDOMINO],
                          [cmn.CELL_SPACE, cmn.CELL_VDOMINO],
                          [cmn.CELL_HDOMINO],
                          [cmn.CELL_VDOMINO, cmn.CELL_SPACE],
                          [cmn.CELL_VDOMINO, cmn.CELL_HDOMINO],
                          [cmn.CELL_VDOMINO, cmn.CELL_VDOMINO]]
    elif num_processors >= 3 and rows >= 3 and cols >= 3:
        num_processors = 3  # Only use 3 processors
        chrom_init_seq = [[cmn.CELL_SPACE],
                          [cmn.CELL_HDOMINO],
                          [cmn.CELL_VDOMINO]]
    else:
        num_processors = 1  # Only use 1 processor
        chrom_init_seq = []

    # Initialise our permutations counter object.
    counter = Counter(0)
    # Setup the BA algorithm to run on num_processors value, if possible.
    if num_processors == 1:
        print("System not sufficient to run parallel processes.")
        ba_proc(rows, cols, chrom_init_seq, 0, counter, calc_permutations)
    else:
        print("Running {} parallel processes.".format(num_processors))
        # Seeding each BA object with all the different possible combinations
        # for the first board squares.
        procs = []
        for i in range(num_processors):
            proc = Process(target=ba_proc, args=(rows, cols, chrom_init_seq[i], i, counter, calc_permutations))
            procs.append(proc)
            proc.start()
        # Block until all cores have finished.
        for proc in procs:
            proc.join()

    execution_time = time.time() - start
    #
    # Now display the number of permutations found.
    #
    if calc_permutations:
        msg = "\n\nNumber of permutations = {}\n".format(counter.value())
        print(msg)
        fname = "{}x{}-permutations.txt".format(rows, cols)
        fname = os.path.join(cmn.PERMUTATIONS_FOLDER, fname)
        with open(fname, "w") as fout:
            fout.write(msg)
    #
    # Parse the solution files; classify the solutions and plot them.
    #
    else:
        # Retrieve the expected number of holes and dominoes
        (holes, dominoes) = cmn.calc_solution_holes(rows, cols)
        dplobj = dpl.CPlot(rows, cols, "BA", holes, dominoes)
        dplobj.load_soln_info()
        dplobj.classify_solns()
        dplobj.plot_all()
        #
	    # Write the version information to the results folder.
	    # - Python and script version information is recorded.
	    # - this includes the execution time in seconds.
	    #
        verstr = cmn.get_version_str(SCRIPTINFO)
        if dplobj.get_total_solns():
            time_1st = float(execution_time / dplobj.get_total_solns())
        else:
            # protect against divide by zero
            time_1st = execution_time
        dplobj.write_test_info(verstr, execution_time, time_1st)
    # Display the time taken if requested.
    if timed_execution:
        print("Script execution time:", execution_time, "seconds")
    # Return the execution time.
    return execution_time
    # Done.


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--version', action='version', version=SCRIPTINFO)
    PARSER.add_argument('grid', nargs=1,
                        help='Gunport problem grid size, eg 10x8.')
    PARSER.add_argument('--permutations', '-p',
                        help='Calculate total permutations.',
                        action='store_true')
    PARSER.add_argument('--timer', '-t',
                        help='Script execution time.',
                        action='store_true')
    # Get the arguments dictionary, where arguments are the keys.
    ARGS = vars(PARSER.parse_args())
    # Extract the grid dimensions as a list [x, y]
    GRID = ARGS['grid']
    GRID = GRID[0].lower().split("x")
    # Set the permutations boolean value
    PERMUTATIONS = ARGS['permutations']
    # Set the timer boolean value
    TIMER = ARGS['timer']

    # Start up the application
    main(GRID, PERMUTATIONS, TIMER)

# EOF

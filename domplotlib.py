#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
Plotting library for dominoes on a grid.
Written for use with the Gunport Problem solving scripts.
"""

import os
import sys
import time                     # Used to time script execution.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.ticker import AutoMinorLocator
import common as cmn            # Common defines and functions
import findallsolns as fas      # Find all solns from fundamental soln (debug)

__author__ = 'David Kind'
__date__ = '11-03-2020'
__version__ = '2.4'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)
# Grid colour definitions
RGB_WHITE = [255, 255, 255]
RGB_RED = [255, 0, 0]
RGB_GREEN = [0, 255, 0]
RGB_BLUE = [0, 0, 255]


class CPlot():
    """
    Gunport Plotting Object.
    """
    def __init__(self, brdrows, brdcols, alg_str, holes, dominoes):
        # Initialise the object members
        # Calculate solution maximum number of holes.
#        (self.holes, self.dominoes) = cmn.calc_solution_holes(brdrows, brdcols)
        # Set the expected number of holes and dominoes
        self.holes = holes
        self.dominoes = dominoes
        # Set the algorithm string currently being used.
        self.alg_str = alg_str
        self.grid_str = "{}x{}".format(brdrows, brdcols)
        self.path_str = ""
        self.title_str = ""
        # Number of total 'all' solutions
        self.total_all_solns = int(0)
        self.total_fundamental = int(0)
        # Set a dummy board full of holes
        self.board = np.zeros((brdrows, brdcols), 'uint8')
        self.rows = brdrows
        self.cols = brdcols
        # files - dictionary solution files; used for characterisation.
        #       {<solution filename>: [<path>:<characterisation string>]}
        self.files = {}
        # processor number refers to the parallel processor processor used
        self.proc_num = 0

    def set_proc_num(self, pnum):
        """
        Sets the processor number value, by default this is zero.
        :param pnum: processor number.
        :return: n/a
        """
        self.proc_num = pnum

    def get_total_solns(self):
        """
        Returns the total number of solutions counted.
        :param: n/a
        :return: integer value of total solutions counted.
        """
        return self.total_all_solns

    def get_total_fundamental(self):
        """
        Returns the total number of solutions counted.
        :param: n/a
        :return: integer value of total solutions counted.
        """
        return self.total_fundamental

    def update(self, brd):
        """
        Updates the class members with the new board and dimensions; these can
        change if the board is rotated to identify additional solutions.
        :param brd: encoded numpy ndarray of dominoes fitted to the grid.
        :return: n/a
        """
        # Get the grid dimension;
        # Note: this can change when the grid is rotated and m != n.
        self.board = brd
        (self.rows, self.cols) = np.shape(brd)
        self.grid_str = "{}x{}".format(self.rows, self.cols)
        grid_path = "__{}_holes-{}_dominoes" \
                    .format(self.holes, self.dominoes)
        grid_path = self.grid_str + grid_path
        # Set the path string to the destination results directory.
        self.path_str = os.path.join(cmn.RESULTS_FOLDER, self.alg_str, grid_path)
        # Make sure to create the directory if not already there.
        if not os.path.exists(self.path_str):
            os.makedirs(self.path_str)
        # Create partial plot title already, to save time.
        # title: "[x]-[y] (m x n) x holes, y dominoes"
        self.title_str = " ({}) {} holes, {} dominoes"\
            .format(self.grid_str, self.holes, self.dominoes)

    def update_results_path(self):
        """
        Updates the class members with the new board and dimensions; these can
        change if the board is rotated to identify additional solutions.
        :param:  n/a
        :return: n/a
        """
        # Get the board dimension;
        # Note: this can change when the board is rotated and m != n.
        (self.rows, self.cols) = np.shape(self.board)
        self.grid_str = "{}x{}".format(self.rows, self.cols)
        grid_path = "__{}_holes-{}_dominoes" \
                    .format(self.holes, self.dominoes)
        grid_path = self.grid_str + grid_path
        # Set the path string to the destination results directory.
        self.path_str = os.path.join(cmn.RESULTS_FOLDER, self.alg_str, grid_path)
        # Make sure to create the directory if not already there.
        if not os.path.exists(self.path_str):
            os.makedirs(self.path_str)

    def write_soln_to_file(self, brd):
        """
        Writes the solution to a text file; this means the solutions can be
        processed again, as and when required.
        :param:  brd - numpy ndarray or board.
        :return: n/a
        """
        self.board = brd
        self.update_results_path()  # Board may have changed orientation
        fname = "soln-{}-{}.npy".format(self.proc_num, self.total_all_solns)
        fsolution = os.path.join(self.path_str, fname)
        # Write the solution to the specified file.
        print("Writing solution: {}".format(fsolution))
        np.save(fsolution, brd, allow_pickle=False, fix_imports=False)
        # Add the details to the files dictionary.
        self.files[fname] = [self.path_str, None]
        # Be sure to update the solution number total
        self.total_all_solns = self.total_all_solns + 1

    def load_soln_info(self):
        """
        Used to help re-classify a results directory, which already has all
        the solutions written to it. The class variable members need to be
        updated with the number of solutions and the solution file names.
        :param:  n/a
        :return: n/a
        """
        self.update(self.board)
        # Scan results directory for soln-<num>.npy files
        for fname in os.listdir(self.path_str):
            if fname.endswith(".npy"):
                self.files[fname] = [self.path_str, None]
                # Be sure to update the solution number total
                self.total_all_solns = self.total_all_solns + 1

    def classify_solns(self):
        """
        Parses all the solutions files and classifies the solutions as either
        fundamental or as a sub-solution to a fundamental solution.
        Ref: Visio diagram domplotlib.vsd
        :param:  n/a
        :return: n/a
        """
        if not self.total_all_solns:
            # Nothing to do; there are no solutions.
            return
        # Initialise our indexes for fundamental and fundamental sub-solution.
        fidx = int(0)
        # Now process all the recorded solution files.
        for soln in sorted(self.files):
            # Skip if this solution already been classified.
            if self.files[soln][1]:
                continue
            # We have a fundamental solution; update the dictionary.
            self.files[soln][1] = "[{}]-[0]".format(fidx)
            sidx = 1
            # Load the numpy array solution
            fsolution = os.path.join(self.files[soln][0], soln)
            board = np.load(fsolution)
            boards = fas.findall(board)
            # Now parse all the solution files for sub-solutions
            for subsoln in sorted(self.files):
                # Skip if this solution already been classified.
                if self.files[subsoln][1]:
                    continue
                # Load the numpy array solution
                fsolution = os.path.join(self.files[subsoln][0], subsoln)
                board = np.load(fsolution)
#                print(board) # debug
                # Found an unclassified board; is it a sub-solution.
                # Check it against all solution boards.
                for array_board in boards:
                    if np.array_equal(array_board, board):
                        # Found a match; update the dictionary
                        self.files[subsoln][1] = "[{}]-[{}]".format(fidx, sidx)
                        sidx = sidx + 1
                        break
            # MUST increment the fundamental solution count value.
            fidx = fidx + 1
        self.total_fundamental = fidx

    def write_test_info(self, verstr, exetime, time1st):
        """
        Adds the test information text to the results plot folder as a file
        for audit purposes: trace-ability and reproduce-ability.
        :param verstr:  (string)script version test information.
        :param exetime: (float)test total execution time.
        :param time1st: (float)time taken to find 1st solution.
        :return: n/a
        """
        # Add the execution time.
        verstr += " - Execution time: {}s\n".format(exetime)
        # Add the total number of solutions and fundamental solutions
        verstr += " - Total solutions: {}\n".format(self.get_total_solns())
        verstr += " - Fundamental solutions: {}\n" \
            .format(self.get_total_fundamental())
        # Add the solution in dominoes and holes
        verstr += " - Solution: {} holes {} dominoes\n".format(self.holes, self.dominoes)
        # Add the board size (mxn)
        verstr += " - Board: {}x{}\n".format(self.rows, self.cols)
        # Add the time to the 1st solution
        verstr += " - time to 1st solution: {}s".format(time1st)
        # Create the version file for these test results.
        info_fname = os.path.join(self.path_str, cmn.RESULTS_FILE)
        with open(info_fname, "w") as fout:
            fout.write(verstr)

    def convert2rgb(self, grid):
        """
        Converts the 2D numpy ndarray into a an array with colours for
        plotting. The dominoes are colour coded to show orientation and the
        holes are simply shown with no colour.
        Parameters:
            grid:   encoded numpy ndarray of dominoes fitted to the grid
        Returns:
            n/a
        """
        # Add RGB colours to the dominoes in the 2D plot.
        # There are 3 terms required for RGB colouring.
        # Check each square in the grid and add the required colour.
        plotgrid = np.zeros((self.rows, self.cols, 3), 'uint8')
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y, x] == cmn.CELL_SPACE:
                    # Empty Space Colour
                    colour = RGB_WHITE
                elif grid[y, x] == cmn.CELL_HDOMINO:
                    # Horizontal Domino Colour
                    colour = RGB_RED
                else:
                    # grid[y, x] == cmn.CELL_VDOMINO
                    # Vertical Domino Colour
                    colour = RGB_BLUE
                plotgrid[y, x] = colour
        return plotgrid

    def plot_all(self):
        """
        Plot all the solution files according to their classification.
        :param: n/a
        :return: n/a
        """
        # Now process all the recorded solution files.
        for soln in sorted(self.files):
            # Build the numpy array solution file name
            fsolution = os.path.join(self.files[soln][0], soln)
            # Skip if this solution is unclassified.
            if not self.files[soln][1]:
                print("{} is not classified.".format(fsolution))
                continue
            # Load the numpy array solution
            board = np.load(fsolution)
            self.update(board)
            self.plot(self.files[soln][1])

    def plot(self, index_str):
        """
        Displays the location of the holes and dominoes on a grid using the
        matplotlib Python library. The dominoes are colour coded to show
        orientation and the holes are simply shown with no colour.
        Horizontal = Red; Vertical = Blue
        :param:  index_str - solution indexed string
        :return: n/a
        """
        #
        # Convert the 2D array into an RGB equivalent for plotting
        #
        plotgrid = self.convert2rgb(self.board)
        # Now we can set the index string.
#        index_str = "[{}]-[{}]".format(self.findex, self.sindex)
        fname = "{}-{}.png".format(index_str, self.grid_str)
        # Finally put together the solution path.
        soln_path = os.path.join(self.path_str, fname)
        soln_title = "{} {}".format(index_str, self.title_str)
        #
        # Plot the RGB Grid with the colour coded dominoes.
        #
        _, ax = plt.subplots()
        ax.imshow(plotgrid, cmap=cm.hsv, interpolation='None')
        # Change the x-axis tick location to the top of the plot; we're
        # counting rows and columns starting at zero from the top left corner
        # of the grid/board.
        ax.xaxis.tick_top()
        # Configure the major ticks; starting at -0.5 to ensure the grid/board
        # is drawn completely
        ax.set_xticks([x - 0.5 for x in range(self.cols + 1)], minor=False)
        ax.set_yticks([y - 0.5 for y in range(self.rows + 1)], minor=False)
        # Hide the labels for the major ticks on both axis.
        ax.tick_params(axis='both', which='major', labelcolor='white')
        # Set the grid/board line properties
        ax.grid(which='major', linestyle='-', linewidth='1.25', color='black')
        # Configure the minor ticks; we want to use these to number the rows
        # and columns and centre them at the edge of each square. Remember the
        # rows and columns start at the top left corner with at (0, 0).
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        # Add the text labels to the minor ticks for both rows and columns.
        col_labels = [str(x + 1) for x in range(self.cols)]
        ax.set_xticklabels(col_labels, minor=True)
        row_labels = [str(y + 1) for y in range(self.rows)]
        ax.set_yticklabels(row_labels, minor=True)
        # Hide the ticks on both axis; looks better.
        ax.tick_params(axis='both', which='both', length=0.0)
        # Ready to add the title to the result plot
        plt.title(soln_title)
        # Now show/save the resultant plot.
        plt.savefig(soln_path)
#        plt.show()
        # Be sure to clear down the plt object, otherwise is clogs up and
        # eventually runs out of memory.
        plt.close('all')

    def __str__(self):
        # Debug output of internal members
        outstr = "Debug output of members:\n"
        outstr += "Obj Addr={}\n".format(hex(id(self)))
        outstr += "{} holes, {} dominoes\n".format(self.holes, self.dominoes)
        outstr += "Algorithm: {}\n".format(self.alg_str)
        outstr += "grid_str={}\n".format(self.grid_str)
        outstr += "path_str={}\n".format(self.path_str)
        outstr += "title_str={}\n".format(self.title_str)
        outstr += "total_all_solns={}\n".format(self.total_all_solns)
        outstr += "total_fundamental={}\n".format(self.total_fundamental)
        outstr += "board={}\n".format(self.board)
        outstr += "rows={}\n".format(self.rows)
        outstr += "cols={}\n".format(self.cols)
        outstr += "files={}\n".format(self.files)
        return outstr


if __name__ == '__main__':
    START = time.time()        # Used to time script execution.
    ROWS = 3
    COLS = 4
    # Retrieve the expected number of holes and dominoes
    (HOLES, DOMINOES) = cmn.calc_solution_holes(ROWS, COLS)
    DPLOBJ = CPlot(ROWS, COLS, "domplotlib", HOLES, DOMINOES)
    # New solution
    # Fill the board up with holes
    #
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = cmn.CELL_HDOMINO
    BOARD[0, 2] = cmn.CELL_HDOMINO
    BOARD[1, 0] = cmn.CELL_HDOMINO
    BOARD[1, 1] = cmn.CELL_HDOMINO
    BOARD[2, 1] = cmn.CELL_HDOMINO
    BOARD[2, 2] = cmn.CELL_HDOMINO
    BOARD[1, 3] = cmn.CELL_VDOMINO
    BOARD[2, 3] = cmn.CELL_VDOMINO
    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)
    # New solution
    # Fill the board up with holes
    #
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = cmn.CELL_HDOMINO
    BOARD[0, 2] = cmn.CELL_HDOMINO
    BOARD[1, 0] = cmn.CELL_HDOMINO
    BOARD[1, 1] = cmn.CELL_HDOMINO
    BOARD[2, 1] = cmn.CELL_HDOMINO
    BOARD[2, 2] = cmn.CELL_HDOMINO
    BOARD[1, 2] = cmn.CELL_HDOMINO
    BOARD[1, 3] = cmn.CELL_HDOMINO
    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)
    # New solution
    # Fill the board up with holes
    #
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[1, 0] = cmn.CELL_VDOMINO
    BOARD[2, 0] = cmn.CELL_VDOMINO
    BOARD[0, 1] = cmn.CELL_VDOMINO
    BOARD[1, 1] = cmn.CELL_VDOMINO
    BOARD[1, 2] = cmn.CELL_VDOMINO
    BOARD[2, 2] = cmn.CELL_VDOMINO
    BOARD[0, 3] = cmn.CELL_VDOMINO
    BOARD[1, 3] = cmn.CELL_VDOMINO
    # Generate all the solutions for this BOARD, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)
    # Determine the execution time.
    EXECUTION_TIME = time.time() - START
    TIME_1ST = float(EXECUTION_TIME / len(ALL_SOLNS))
    #
    # Parse the solution files; classify the solutions and plot them.
    #
    DPLOBJ.classify_solns()
    DPLOBJ.plot_all()
    #
    # Write the version information to the results folder.
    # - Python and script version information is recorded.
    # - this includes the execution time in seconds.
    #
    VERSTR = cmn.get_version_str(SCRIPTINFO)
    DPLOBJ.write_test_info(VERSTR, EXECUTION_TIME, TIME_1ST)

    # ####################################################################
    # (10 x 8) Solutions
    # Huval's Solution
    #
    ROWS = 10
    COLS = 8
    # Retrieve the expected number of holes and DOMINOES
    (HOLES, DOMINOES) = cmn.calc_solution_holes(ROWS, COLS)
    DPLOBJ = CPlot(ROWS, COLS, "domplotlib", HOLES, DOMINOES)
    #
    # Fill the board up with holes
    #
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = cmn.CELL_VDOMINO
    BOARD[0, 3] = cmn.CELL_HDOMINO
    BOARD[0, 4] = cmn.CELL_HDOMINO
    BOARD[0, 6] = cmn.CELL_VDOMINO

    BOARD[1, 0] = cmn.CELL_VDOMINO
    BOARD[1, 1] = cmn.CELL_VDOMINO
    BOARD[1, 2] = cmn.CELL_HDOMINO
    BOARD[1, 3] = cmn.CELL_HDOMINO
    BOARD[1, 5] = cmn.CELL_VDOMINO
    BOARD[1, 6] = cmn.CELL_VDOMINO
    BOARD[1, 7] = cmn.CELL_VDOMINO

    BOARD[2, 0] = cmn.CELL_VDOMINO
    BOARD[2, 2] = cmn.CELL_VDOMINO
    BOARD[2, 4] = cmn.CELL_VDOMINO
    BOARD[2, 5] = cmn.CELL_VDOMINO
    BOARD[2, 7] = cmn.CELL_VDOMINO

    BOARD[3, 1] = cmn.CELL_VDOMINO
    BOARD[3, 2] = cmn.CELL_VDOMINO
    BOARD[3, 3] = cmn.CELL_VDOMINO
    BOARD[3, 4] = cmn.CELL_VDOMINO
    BOARD[3, 6] = cmn.CELL_VDOMINO

    BOARD[4, 0] = cmn.CELL_VDOMINO
    BOARD[4, 1] = cmn.CELL_VDOMINO
    BOARD[4, 3] = cmn.CELL_VDOMINO
    BOARD[4, 5] = cmn.CELL_VDOMINO
    BOARD[4, 6] = cmn.CELL_VDOMINO
    BOARD[4, 7] = cmn.CELL_VDOMINO

    BOARD[5, 0] = cmn.CELL_VDOMINO
    BOARD[5, 2] = cmn.CELL_VDOMINO
    BOARD[5, 4] = cmn.CELL_VDOMINO
    BOARD[5, 5] = cmn.CELL_VDOMINO
    BOARD[5, 7] = cmn.CELL_VDOMINO

    BOARD[6, 1] = cmn.CELL_VDOMINO
    BOARD[6, 2] = cmn.CELL_VDOMINO
    BOARD[6, 3] = cmn.CELL_VDOMINO
    BOARD[6, 4] = cmn.CELL_VDOMINO
    BOARD[6, 6] = cmn.CELL_VDOMINO

    BOARD[7, 0] = cmn.CELL_VDOMINO
    BOARD[7, 1] = cmn.CELL_VDOMINO
    BOARD[7, 3] = cmn.CELL_VDOMINO
    BOARD[7, 5] = cmn.CELL_VDOMINO
    BOARD[7, 6] = cmn.CELL_VDOMINO
    BOARD[7, 7] = cmn.CELL_VDOMINO

    BOARD[8, 0] = cmn.CELL_VDOMINO
    BOARD[8, 2] = cmn.CELL_HDOMINO
    BOARD[8, 3] = cmn.CELL_HDOMINO
    BOARD[8, 4] = cmn.CELL_VDOMINO
    BOARD[8, 5] = cmn.CELL_VDOMINO
    BOARD[8, 7] = cmn.CELL_VDOMINO

    BOARD[9, 1] = cmn.CELL_HDOMINO
    BOARD[9, 2] = cmn.CELL_HDOMINO
    BOARD[9, 4] = cmn.CELL_VDOMINO
    BOARD[9, 5] = cmn.CELL_HDOMINO
    BOARD[9, 6] = cmn.CELL_HDOMINO

    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)
    #
    # Pearce's Solution #1 (8 x 10)
    #
    # Fill the board up with holes
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = cmn.CELL_VDOMINO
    BOARD[0, 2] = cmn.CELL_HDOMINO
    BOARD[0, 3] = cmn.CELL_HDOMINO
    BOARD[0, 5] = cmn.CELL_HDOMINO
    BOARD[0, 6] = cmn.CELL_HDOMINO

    BOARD[1, 0] = cmn.CELL_VDOMINO
    BOARD[1, 1] = cmn.CELL_VDOMINO
    BOARD[1, 3] = cmn.CELL_HDOMINO
    BOARD[1, 4] = cmn.CELL_HDOMINO
    BOARD[1, 6] = cmn.CELL_HDOMINO
    BOARD[1, 7] = cmn.CELL_HDOMINO

    BOARD[2, 0] = cmn.CELL_VDOMINO
    BOARD[2, 2] = cmn.CELL_HDOMINO
    BOARD[2, 3] = cmn.CELL_HDOMINO
    BOARD[2, 5] = cmn.CELL_HDOMINO
    BOARD[2, 6] = cmn.CELL_HDOMINO

    BOARD[3, 1] = cmn.CELL_HDOMINO
    BOARD[3, 2] = cmn.CELL_HDOMINO
    BOARD[3, 4] = cmn.CELL_HDOMINO
    BOARD[3, 5] = cmn.CELL_HDOMINO
    BOARD[3, 7] = cmn.CELL_VDOMINO

    BOARD[4, 0] = cmn.CELL_HDOMINO
    BOARD[4, 1] = cmn.CELL_HDOMINO
    BOARD[4, 3] = cmn.CELL_HDOMINO
    BOARD[4, 4] = cmn.CELL_HDOMINO
    BOARD[4, 6] = cmn.CELL_VDOMINO
    BOARD[4, 7] = cmn.CELL_VDOMINO

    BOARD[5, 1] = cmn.CELL_HDOMINO
    BOARD[5, 2] = cmn.CELL_HDOMINO
    BOARD[5, 4] = cmn.CELL_HDOMINO
    BOARD[5, 5] = cmn.CELL_HDOMINO
    BOARD[5, 6] = cmn.CELL_VDOMINO

    BOARD[6, 0] = cmn.CELL_HDOMINO
    BOARD[6, 1] = cmn.CELL_HDOMINO
    BOARD[6, 3] = cmn.CELL_HDOMINO
    BOARD[6, 4] = cmn.CELL_HDOMINO
    BOARD[6, 6] = cmn.CELL_HDOMINO
    BOARD[6, 7] = cmn.CELL_HDOMINO

    BOARD[7, 1] = cmn.CELL_HDOMINO
    BOARD[7, 2] = cmn.CELL_HDOMINO
    BOARD[7, 4] = cmn.CELL_HDOMINO
    BOARD[7, 5] = cmn.CELL_HDOMINO
    BOARD[7, 7] = cmn.CELL_VDOMINO

    BOARD[8, 0] = cmn.CELL_HDOMINO
    BOARD[8, 1] = cmn.CELL_HDOMINO
    BOARD[8, 3] = cmn.CELL_HDOMINO
    BOARD[8, 4] = cmn.CELL_HDOMINO
    BOARD[8, 6] = cmn.CELL_VDOMINO
    BOARD[8, 7] = cmn.CELL_VDOMINO

    BOARD[9, 1] = cmn.CELL_HDOMINO
    BOARD[9, 2] = cmn.CELL_HDOMINO
    BOARD[9, 4] = cmn.CELL_HDOMINO
    BOARD[9, 5] = cmn.CELL_HDOMINO
    BOARD[9, 6] = cmn.CELL_VDOMINO

    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)
    #
    # Pearce's Solution #2 (8 x 10)
    #
    # Fill the board up with holes
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = cmn.CELL_HDOMINO
    BOARD[0, 2] = cmn.CELL_HDOMINO
    BOARD[0, 4] = cmn.CELL_HDOMINO
    BOARD[0, 5] = cmn.CELL_HDOMINO
    BOARD[0, 7] = cmn.CELL_VDOMINO

    BOARD[1, 0] = cmn.CELL_HDOMINO
    BOARD[1, 1] = cmn.CELL_HDOMINO
    BOARD[1, 3] = cmn.CELL_HDOMINO
    BOARD[1, 4] = cmn.CELL_HDOMINO
    BOARD[1, 6] = cmn.CELL_VDOMINO
    BOARD[1, 7] = cmn.CELL_VDOMINO

    BOARD[2, 1] = cmn.CELL_HDOMINO
    BOARD[2, 2] = cmn.CELL_HDOMINO
    BOARD[2, 4] = cmn.CELL_HDOMINO
    BOARD[2, 5] = cmn.CELL_HDOMINO
    BOARD[2, 6] = cmn.CELL_VDOMINO

    BOARD[3, 0] = cmn.CELL_HDOMINO
    BOARD[3, 1] = cmn.CELL_HDOMINO
    BOARD[3, 3] = cmn.CELL_HDOMINO
    BOARD[3, 4] = cmn.CELL_HDOMINO
    BOARD[3, 6] = cmn.CELL_HDOMINO
    BOARD[3, 7] = cmn.CELL_HDOMINO

    BOARD[4, 1] = cmn.CELL_VDOMINO
    BOARD[4, 2] = cmn.CELL_HDOMINO
    BOARD[4, 3] = cmn.CELL_HDOMINO
    BOARD[4, 5] = cmn.CELL_HDOMINO
    BOARD[4, 6] = cmn.CELL_HDOMINO

    BOARD[5, 0] = cmn.CELL_VDOMINO
    BOARD[5, 1] = cmn.CELL_VDOMINO
    BOARD[5, 3] = cmn.CELL_HDOMINO
    BOARD[5, 4] = cmn.CELL_HDOMINO
    BOARD[5, 6] = cmn.CELL_HDOMINO
    BOARD[5, 7] = cmn.CELL_HDOMINO

    BOARD[6, 0] = cmn.CELL_VDOMINO
    BOARD[6, 2] = cmn.CELL_HDOMINO
    BOARD[6, 3] = cmn.CELL_HDOMINO
    BOARD[6, 5] = cmn.CELL_HDOMINO
    BOARD[6, 6] = cmn.CELL_HDOMINO

    BOARD[7, 1] = cmn.CELL_HDOMINO
    BOARD[7, 2] = cmn.CELL_HDOMINO
    BOARD[7, 4] = cmn.CELL_HDOMINO
    BOARD[7, 5] = cmn.CELL_HDOMINO
    BOARD[7, 7] = cmn.CELL_VDOMINO

    BOARD[8, 0] = cmn.CELL_HDOMINO
    BOARD[8, 1] = cmn.CELL_HDOMINO
    BOARD[8, 3] = cmn.CELL_HDOMINO
    BOARD[8, 4] = cmn.CELL_HDOMINO
    BOARD[8, 6] = cmn.CELL_VDOMINO
    BOARD[8, 7] = cmn.CELL_VDOMINO

    BOARD[9, 1] = cmn.CELL_HDOMINO
    BOARD[9, 2] = cmn.CELL_HDOMINO
    BOARD[9, 4] = cmn.CELL_HDOMINO
    BOARD[9, 5] = cmn.CELL_HDOMINO
    BOARD[9, 6] = cmn.CELL_VDOMINO

    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)

    #
    # Solution to (13 x 14)
    #
    ROWS = 13
    COLS = 14
    # Retrieve the expected number of holes and DOMINOES
    (HOLES, DOMINOES) = cmn.calc_solution_holes(ROWS, COLS)
    DPLOBJ = CPlot(ROWS, COLS, "domplotlib", HOLES, DOMINOES)
    # Fill the board up with holes
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = cmn.CELL_HDOMINO
    BOARD[0, 2] = cmn.CELL_HDOMINO
    BOARD[0, 4] = cmn.CELL_HDOMINO
    BOARD[0, 5] = cmn.CELL_HDOMINO
    BOARD[0, 7] = cmn.CELL_HDOMINO
    BOARD[0, 8] = cmn.CELL_HDOMINO
    BOARD[0, 10] = cmn.CELL_HDOMINO
    BOARD[0, 11] = cmn.CELL_HDOMINO
    BOARD[0, 12] = cmn.CELL_VDOMINO

    BOARD[1, 0] = cmn.CELL_HDOMINO
    BOARD[1, 1] = cmn.CELL_HDOMINO
    BOARD[1, 3] = cmn.CELL_HDOMINO
    BOARD[1, 4] = cmn.CELL_HDOMINO
    BOARD[1, 6] = cmn.CELL_HDOMINO
    BOARD[1, 7] = cmn.CELL_HDOMINO
    BOARD[1, 9] = cmn.CELL_HDOMINO
    BOARD[1, 10] = cmn.CELL_HDOMINO
    BOARD[1, 12] = cmn.CELL_VDOMINO
    BOARD[1, 13] = cmn.CELL_VDOMINO

    BOARD[2, 1] = cmn.CELL_HDOMINO
    BOARD[2, 2] = cmn.CELL_HDOMINO
    BOARD[2, 4] = cmn.CELL_HDOMINO
    BOARD[2, 5] = cmn.CELL_HDOMINO
    BOARD[2, 7] = cmn.CELL_HDOMINO
    BOARD[2, 8] = cmn.CELL_HDOMINO
    BOARD[2, 10] = cmn.CELL_HDOMINO
    BOARD[2, 11] = cmn.CELL_HDOMINO
    BOARD[2, 13] = cmn.CELL_VDOMINO

    BOARD[3, 0] = cmn.CELL_VDOMINO
    BOARD[3, 2] = cmn.CELL_HDOMINO
    BOARD[3, 3] = cmn.CELL_HDOMINO
    BOARD[3, 5] = cmn.CELL_HDOMINO
    BOARD[3, 6] = cmn.CELL_HDOMINO
    BOARD[3, 8] = cmn.CELL_HDOMINO
    BOARD[3, 9] = cmn.CELL_HDOMINO
    BOARD[3, 11] = cmn.CELL_HDOMINO
    BOARD[3, 12] = cmn.CELL_HDOMINO

    BOARD[4, 0] = cmn.CELL_VDOMINO
    BOARD[4, 1] = cmn.CELL_VDOMINO
    BOARD[4, 3] = cmn.CELL_HDOMINO
    BOARD[4, 4] = cmn.CELL_HDOMINO
    BOARD[4, 6] = cmn.CELL_HDOMINO
    BOARD[4, 7] = cmn.CELL_HDOMINO
    BOARD[4, 9] = cmn.CELL_HDOMINO
    BOARD[4, 10] = cmn.CELL_HDOMINO
    BOARD[4, 12] = cmn.CELL_HDOMINO
    BOARD[4, 13] = cmn.CELL_HDOMINO

    BOARD[5, 1] = cmn.CELL_VDOMINO
    BOARD[5, 2] = cmn.CELL_HDOMINO
    BOARD[5, 3] = cmn.CELL_HDOMINO
    BOARD[5, 5] = cmn.CELL_HDOMINO
    BOARD[5, 6] = cmn.CELL_HDOMINO
    BOARD[5, 8] = cmn.CELL_HDOMINO
    BOARD[5, 9] = cmn.CELL_HDOMINO
    BOARD[5, 11] = cmn.CELL_HDOMINO
    BOARD[5, 12] = cmn.CELL_HDOMINO

    BOARD[6, 0] = cmn.CELL_HDOMINO
    BOARD[6, 1] = cmn.CELL_HDOMINO
    BOARD[6, 3] = cmn.CELL_HDOMINO
    BOARD[6, 4] = cmn.CELL_HDOMINO
    BOARD[6, 6] = cmn.CELL_HDOMINO
    BOARD[6, 7] = cmn.CELL_HDOMINO
    BOARD[6, 9] = cmn.CELL_HDOMINO
    BOARD[6, 10] = cmn.CELL_HDOMINO
    BOARD[6, 12] = cmn.CELL_HDOMINO
    BOARD[6, 13] = cmn.CELL_HDOMINO

    BOARD[7, 1] = cmn.CELL_HDOMINO
    BOARD[7, 2] = cmn.CELL_HDOMINO
    BOARD[7, 4] = cmn.CELL_HDOMINO
    BOARD[7, 5] = cmn.CELL_HDOMINO
    BOARD[7, 7] = cmn.CELL_HDOMINO
    BOARD[7, 8] = cmn.CELL_HDOMINO
    BOARD[7, 10] = cmn.CELL_HDOMINO
    BOARD[7, 11] = cmn.CELL_HDOMINO
    BOARD[7, 12] = cmn.CELL_VDOMINO

    BOARD[8, 0] = cmn.CELL_HDOMINO
    BOARD[8, 1] = cmn.CELL_HDOMINO
    BOARD[8, 3] = cmn.CELL_HDOMINO
    BOARD[8, 4] = cmn.CELL_HDOMINO
    BOARD[8, 6] = cmn.CELL_HDOMINO
    BOARD[8, 7] = cmn.CELL_HDOMINO
    BOARD[8, 9] = cmn.CELL_HDOMINO
    BOARD[8, 10] = cmn.CELL_HDOMINO
    BOARD[8, 12] = cmn.CELL_VDOMINO
    BOARD[8, 13] = cmn.CELL_VDOMINO

    BOARD[9, 1] = cmn.CELL_HDOMINO
    BOARD[9, 2] = cmn.CELL_HDOMINO
    BOARD[9, 4] = cmn.CELL_HDOMINO
    BOARD[9, 5] = cmn.CELL_HDOMINO
    BOARD[9, 7] = cmn.CELL_HDOMINO
    BOARD[9, 8] = cmn.CELL_HDOMINO
    BOARD[9, 10] = cmn.CELL_HDOMINO
    BOARD[9, 11] = cmn.CELL_HDOMINO
    BOARD[9, 13] = cmn.CELL_VDOMINO

    BOARD[10, 0] = cmn.CELL_VDOMINO
    BOARD[10, 2] = cmn.CELL_HDOMINO
    BOARD[10, 3] = cmn.CELL_HDOMINO
    BOARD[10, 5] = cmn.CELL_HDOMINO
    BOARD[10, 6] = cmn.CELL_HDOMINO
    BOARD[10, 8] = cmn.CELL_HDOMINO
    BOARD[10, 9] = cmn.CELL_HDOMINO
    BOARD[10, 11] = cmn.CELL_HDOMINO
    BOARD[10, 12] = cmn.CELL_HDOMINO

    BOARD[11, 0] = cmn.CELL_VDOMINO
    BOARD[11, 1] = cmn.CELL_VDOMINO
    BOARD[11, 3] = cmn.CELL_HDOMINO
    BOARD[11, 4] = cmn.CELL_HDOMINO
    BOARD[11, 6] = cmn.CELL_HDOMINO
    BOARD[11, 7] = cmn.CELL_HDOMINO
    BOARD[11, 9] = cmn.CELL_HDOMINO
    BOARD[11, 10] = cmn.CELL_HDOMINO
    BOARD[11, 12] = cmn.CELL_HDOMINO
    BOARD[11, 13] = cmn.CELL_HDOMINO

    BOARD[12, 1] = cmn.CELL_VDOMINO
    BOARD[12, 2] = cmn.CELL_HDOMINO
    BOARD[12, 3] = cmn.CELL_HDOMINO
    BOARD[12, 5] = cmn.CELL_HDOMINO
    BOARD[12, 6] = cmn.CELL_HDOMINO
    BOARD[12, 8] = cmn.CELL_HDOMINO
    BOARD[12, 9] = cmn.CELL_HDOMINO
    BOARD[12, 11] = cmn.CELL_HDOMINO
    BOARD[12, 12] = cmn.CELL_HDOMINO

    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)

    #
    # Solution to (16 x 14)
    #
    ROWS = 16
    COLS = 14
    # Retrieve the expected number of holes and DOMINOES
    (HOLES, DOMINOES) = cmn.calc_solution_holes(ROWS, COLS)
    DPLOBJ = CPlot(ROWS, COLS, "domplotlib", HOLES, DOMINOES)
    # Fill the board up with holes
    BOARD = np.zeros((ROWS, COLS), 'uint8')
    # Add a new fundamental shape to the test [row, col]
    BOARD[0, 1] = cmn.CELL_HDOMINO
    BOARD[0, 2] = cmn.CELL_HDOMINO
    BOARD[0, 4] = cmn.CELL_HDOMINO
    BOARD[0, 5] = cmn.CELL_HDOMINO
    BOARD[0, 7] = cmn.CELL_HDOMINO
    BOARD[0, 8] = cmn.CELL_HDOMINO
    BOARD[0, 10] = cmn.CELL_HDOMINO
    BOARD[0, 11] = cmn.CELL_HDOMINO
    BOARD[0, 12] = cmn.CELL_VDOMINO

    BOARD[1, 0] = cmn.CELL_HDOMINO
    BOARD[1, 1] = cmn.CELL_HDOMINO
    BOARD[1, 3] = cmn.CELL_HDOMINO
    BOARD[1, 4] = cmn.CELL_HDOMINO
    BOARD[1, 6] = cmn.CELL_HDOMINO
    BOARD[1, 7] = cmn.CELL_HDOMINO
    BOARD[1, 9] = cmn.CELL_HDOMINO
    BOARD[1, 10] = cmn.CELL_HDOMINO
    BOARD[1, 12] = cmn.CELL_VDOMINO
    BOARD[1, 13] = cmn.CELL_VDOMINO

    BOARD[2, 1] = cmn.CELL_HDOMINO
    BOARD[2, 2] = cmn.CELL_HDOMINO
    BOARD[2, 4] = cmn.CELL_HDOMINO
    BOARD[2, 5] = cmn.CELL_HDOMINO
    BOARD[2, 7] = cmn.CELL_HDOMINO
    BOARD[2, 8] = cmn.CELL_HDOMINO
    BOARD[2, 10] = cmn.CELL_HDOMINO
    BOARD[2, 11] = cmn.CELL_HDOMINO
    BOARD[2, 13] = cmn.CELL_VDOMINO

    BOARD[3, 0] = cmn.CELL_VDOMINO
    BOARD[3, 2] = cmn.CELL_HDOMINO
    BOARD[3, 3] = cmn.CELL_HDOMINO
    BOARD[3, 5] = cmn.CELL_HDOMINO
    BOARD[3, 6] = cmn.CELL_HDOMINO
    BOARD[3, 8] = cmn.CELL_HDOMINO
    BOARD[3, 9] = cmn.CELL_HDOMINO
    BOARD[3, 11] = cmn.CELL_HDOMINO
    BOARD[3, 12] = cmn.CELL_HDOMINO

    BOARD[4, 0] = cmn.CELL_VDOMINO
    BOARD[4, 1] = cmn.CELL_VDOMINO
    BOARD[4, 3] = cmn.CELL_HDOMINO
    BOARD[4, 4] = cmn.CELL_HDOMINO
    BOARD[4, 6] = cmn.CELL_HDOMINO
    BOARD[4, 7] = cmn.CELL_HDOMINO
    BOARD[4, 9] = cmn.CELL_HDOMINO
    BOARD[4, 10] = cmn.CELL_HDOMINO
    BOARD[4, 12] = cmn.CELL_HDOMINO
    BOARD[4, 13] = cmn.CELL_HDOMINO

    BOARD[5, 1] = cmn.CELL_VDOMINO
    BOARD[5, 2] = cmn.CELL_HDOMINO
    BOARD[5, 3] = cmn.CELL_HDOMINO
    BOARD[5, 5] = cmn.CELL_HDOMINO
    BOARD[5, 6] = cmn.CELL_HDOMINO
    BOARD[5, 8] = cmn.CELL_HDOMINO
    BOARD[5, 9] = cmn.CELL_HDOMINO
    BOARD[5, 11] = cmn.CELL_HDOMINO
    BOARD[5, 12] = cmn.CELL_HDOMINO

    BOARD[6, 0] = cmn.CELL_HDOMINO
    BOARD[6, 1] = cmn.CELL_HDOMINO
    BOARD[6, 3] = cmn.CELL_HDOMINO
    BOARD[6, 4] = cmn.CELL_HDOMINO
    BOARD[6, 6] = cmn.CELL_HDOMINO
    BOARD[6, 7] = cmn.CELL_HDOMINO
    BOARD[6, 9] = cmn.CELL_HDOMINO
    BOARD[6, 10] = cmn.CELL_HDOMINO
    BOARD[6, 12] = cmn.CELL_HDOMINO
    BOARD[6, 13] = cmn.CELL_HDOMINO

    BOARD[7, 1] = cmn.CELL_HDOMINO
    BOARD[7, 2] = cmn.CELL_HDOMINO
    BOARD[7, 4] = cmn.CELL_HDOMINO
    BOARD[7, 5] = cmn.CELL_HDOMINO
    BOARD[7, 7] = cmn.CELL_HDOMINO
    BOARD[7, 8] = cmn.CELL_HDOMINO
    BOARD[7, 10] = cmn.CELL_HDOMINO
    BOARD[7, 11] = cmn.CELL_HDOMINO
    BOARD[7, 12] = cmn.CELL_VDOMINO

    BOARD[8, 0] = cmn.CELL_HDOMINO
    BOARD[8, 1] = cmn.CELL_HDOMINO
    BOARD[8, 3] = cmn.CELL_HDOMINO
    BOARD[8, 4] = cmn.CELL_HDOMINO
    BOARD[8, 6] = cmn.CELL_HDOMINO
    BOARD[8, 7] = cmn.CELL_HDOMINO
    BOARD[8, 9] = cmn.CELL_HDOMINO
    BOARD[8, 10] = cmn.CELL_HDOMINO
    BOARD[8, 12] = cmn.CELL_VDOMINO
    BOARD[8, 13] = cmn.CELL_VDOMINO

    BOARD[9, 1] = cmn.CELL_HDOMINO
    BOARD[9, 2] = cmn.CELL_HDOMINO
    BOARD[9, 4] = cmn.CELL_HDOMINO
    BOARD[9, 5] = cmn.CELL_HDOMINO
    BOARD[9, 7] = cmn.CELL_HDOMINO
    BOARD[9, 8] = cmn.CELL_HDOMINO
    BOARD[9, 10] = cmn.CELL_HDOMINO
    BOARD[9, 11] = cmn.CELL_HDOMINO
    BOARD[9, 13] = cmn.CELL_VDOMINO

    BOARD[10, 0] = cmn.CELL_VDOMINO
    BOARD[10, 2] = cmn.CELL_HDOMINO
    BOARD[10, 3] = cmn.CELL_HDOMINO
    BOARD[10, 5] = cmn.CELL_HDOMINO
    BOARD[10, 6] = cmn.CELL_HDOMINO
    BOARD[10, 8] = cmn.CELL_HDOMINO
    BOARD[10, 9] = cmn.CELL_HDOMINO
    BOARD[10, 11] = cmn.CELL_HDOMINO
    BOARD[10, 12] = cmn.CELL_HDOMINO

    BOARD[11, 0] = cmn.CELL_VDOMINO
    BOARD[11, 1] = cmn.CELL_VDOMINO
    BOARD[11, 3] = cmn.CELL_HDOMINO
    BOARD[11, 4] = cmn.CELL_HDOMINO
    BOARD[11, 6] = cmn.CELL_HDOMINO
    BOARD[11, 7] = cmn.CELL_HDOMINO
    BOARD[11, 9] = cmn.CELL_HDOMINO
    BOARD[11, 10] = cmn.CELL_HDOMINO
    BOARD[11, 12] = cmn.CELL_HDOMINO
    BOARD[11, 13] = cmn.CELL_HDOMINO

    BOARD[12, 0] = cmn.CELL_VDOMINO
    BOARD[12, 1] = cmn.CELL_VDOMINO
    BOARD[12, 2] = cmn.CELL_HDOMINO
    BOARD[12, 3] = cmn.CELL_HDOMINO
    BOARD[12, 5] = cmn.CELL_HDOMINO
    BOARD[12, 6] = cmn.CELL_HDOMINO
    BOARD[12, 8] = cmn.CELL_HDOMINO
    BOARD[12, 9] = cmn.CELL_HDOMINO
    BOARD[12, 11] = cmn.CELL_HDOMINO
    BOARD[12, 12] = cmn.CELL_HDOMINO

    BOARD[13, 0] = cmn.CELL_VDOMINO
    BOARD[13, 1] = cmn.CELL_HDOMINO
    BOARD[13, 2] = cmn.CELL_HDOMINO
    BOARD[13, 4] = cmn.CELL_HDOMINO
    BOARD[13, 5] = cmn.CELL_HDOMINO
    BOARD[13, 7] = cmn.CELL_HDOMINO
    BOARD[13, 8] = cmn.CELL_HDOMINO
    BOARD[13, 10] = cmn.CELL_HDOMINO
    BOARD[13, 11] = cmn.CELL_HDOMINO
    BOARD[13, 13] = cmn.CELL_VDOMINO

    BOARD[14, 0] = cmn.CELL_HDOMINO
    BOARD[14, 1] = cmn.CELL_HDOMINO
    BOARD[14, 3] = cmn.CELL_HDOMINO
    BOARD[14, 4] = cmn.CELL_HDOMINO
    BOARD[14, 6] = cmn.CELL_HDOMINO
    BOARD[14, 7] = cmn.CELL_HDOMINO
    BOARD[14, 9] = cmn.CELL_HDOMINO
    BOARD[14, 10] = cmn.CELL_HDOMINO
    BOARD[14, 12] = cmn.CELL_VDOMINO
    BOARD[14, 13] = cmn.CELL_VDOMINO

    BOARD[15, 1] = cmn.CELL_HDOMINO
    BOARD[15, 2] = cmn.CELL_HDOMINO
    BOARD[15, 4] = cmn.CELL_HDOMINO
    BOARD[15, 5] = cmn.CELL_HDOMINO
    BOARD[15, 7] = cmn.CELL_HDOMINO
    BOARD[15, 8] = cmn.CELL_HDOMINO
    BOARD[15, 10] = cmn.CELL_HDOMINO
    BOARD[15, 11] = cmn.CELL_HDOMINO
    BOARD[15, 12] = cmn.CELL_VDOMINO

    # Generate all the solutions for this board, make life easier.
    print("this is the fundamental solution.")
    print(BOARD)
    print("These are the resulting solutions")
    ALL_SOLNS = fas.findall(BOARD)
    for solution in ALL_SOLNS:
        print(solution)
        DPLOBJ.write_soln_to_file(solution)

    # Determine the execution time.
    EXECUTION_TIME = time.time() - START
    TIME_1ST = float(EXECUTION_TIME / len(ALL_SOLNS))
    #
    # Parse the solution files; classify the solutions and plot them.
    #
    DPLOBJ.classify_solns()
    DPLOBJ.plot_all()
    #
    # Write the version information to the results folder.
    # - Python and script version information is recorded.
    # - this includes the execution time in seconds.
    #
    VERSTR = cmn.get_version_str(SCRIPTINFO)
    DPLOBJ.write_test_info(VERSTR, EXECUTION_TIME, TIME_1ST)

# EOF

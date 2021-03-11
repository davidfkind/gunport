#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#

"""
Helper script for the Gunport problem dissertation.
Parses all the results, under the results folder, to produce the dissertation
results graphs and tables. This is a stand-alone script.

Command line usage example:
    py -3 results.py
    py -3 results.py --version
"""

import os
import sys
#import copy
import argparse
import re
import texttable as tt
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.ticker import AutoMinorLocator
import common as cmn            # Common defines and functions


__author__ = 'David Kind'
__date__ = '17-02-2020'
__version__ = '2.3'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)

# Solutions dictionary list index defines
I_PYVERSION = 0
I_OS = 1
I_EXETIME = 2
I_ALLSOLNS = 3
I_FUNSOLNS = 4
I_SOLNSTR = 5
# Note: don't need to store the board dimensions here.
I_1STTIME = 6

class CResults():
    """
    Gunport Results Object.
    """
    def __init__(self, folder):
        """
        Initialises the results object and checks to see if the specified
        results folder exists and if it contains any valid data.
        :param folder: (string)folder name to be tested.
        :return: n/a
        """
        #
        # Initialise the class members...
        #
        self.results_folders = []   # solutions folder list
        self.algorithm = ""         # name of algorithm being processed.
        # Solutions dictionary is formatted as follows:
        # The board size is used as the key for each entry.
        # ie. {<mxn>:[<Python version>, <OS>, <execution time>, ...]}
        # where:
        #   <mxn>:                   (string)board dimensions (key).
        #   <Python version>:        (string)Version of Python the script was
        #                            executed with.
        #   <OS>:                    (string)OS script was executed on.
        #   <execution time>:        (float)time taken to find all solutions.
        #   <1st solution time>:     (float)time taken to find the first solution.
        #   <total solutions>        (int)total number of solutions found.
        #   <fundamental solutions>: (int)total number of unique solutions found.
        #   <solution>:              (string)solution in terms of holes and
        #                            dominoes.
        self.solutions = {}
        # Extract the list of results folders in the specified folder
        # Ensure the results folder actually exists.
        if os.path.exists(folder):
            # The results folder exists; now verify that it contains any results
            # data. Scan the folder for sub-folders containing actual results.
            dirlist = [name for name in os.listdir(folder) \
                      if os.path.isdir(os.path.join(folder, name))]
            for fdir in dirlist:
                if os.path.isfile(os.path.join(folder, fdir, cmn.RESULTS_FILE)):
                    self.results_folders.append(os.path.join(folder, fdir))
        #
        # Extract the algorithm name string.
        #
        self.algorithm = os.path.split(folder)[1]

    def empty(self):
        """
        True if the results are empty else False.
        :return: True if the results folder is empty.
        """
        return not self.results_folders

    def extract_solns(self):
        """
        Extracts the test result information from the results text file.
        These are then added to a dictionary, where the board size is used as
        the key for each entry.
        ie. {<mxn>:[<Python version>, <OS>, <execution time>, ...]}
        Sets self.solutions a dictionary all solutions found.
        self.results_folders list of folders contain results files.
        :param: n/a
        :return: n/a
        """
        # Define the regex constants; do this before the loop to save time
        regex_key = re.compile(r"Board: (.+)$", re.MULTILINE)
        regex_python = re.compile(r"Python version: (\d+\.\d+\.\d+)", re.MULTILINE)
        regex_os = re.compile(r"Test PC: (.+)$", re.MULTILINE)
        regex_exe = re.compile(r"Execution time:\s+(([0-9]*[.]?[0-9]+([ed][-+]?[0-9]+)?)|(inf)|(nan))s", re.MULTILINE)
#        regex_1st = re.compile(r"time to 1st solution: (\d+\.\d+)s", re.MULTILINE)
        regex_1st = re.compile(r"time to 1st solution:\s+(([0-9]*[.]?[0-9]+([ed][-+]?[0-9]+)?)|(inf)|(nan))s", re.MULTILINE)
        regex_total = re.compile(r"Total solutions: (\d+)", re.MULTILINE)
        regex_fun = re.compile(r"Fundamental solutions: (\d+)", re.MULTILINE)
        regex_soln = re.compile(r"Solution: (.+)$", re.MULTILINE)
        regex_key = re.compile(r"Board: (.+)$", re.MULTILINE)
        # Loop through all the solution files
        for folder in self.results_folders:
            print("Processing folder {}".format(folder))
            # Create the solution path and filename string, so that it can be opened.
            fname = os.path.join(folder, cmn.RESULTS_FILE)
            # Read in the entire results text file
            with open(fname, "r") as fin:
                contents = fin.read()
            # Now extract all the test result information and stuff it into a list, which is ordered.
            dict_values = []
#            print("-" * 80)                                     # debug
#            print("folder={}\nfname={}".format(folder, fname))  # debug
#            print("contents={}".format(contents))               # debug
            soln_key = re.findall(regex_key, contents)[0]
            dict_values.append(re.findall(regex_python, contents)[0])
            dict_values.append(re.findall(regex_os, contents)[0])
            dict_values.append(float(re.findall(regex_exe, contents)[0][0]))
            dict_values.append(int(re.findall(regex_total, contents)[0]))
            dict_values.append(int(re.findall(regex_fun, contents)[0]))
            dict_values.append(re.findall(regex_soln, contents)[0])
            dict_values.append(float(re.findall(regex_1st, contents)[0][0]))
            # Add the results to the dictionary
            self.solutions[soln_key] = dict_values
        ### Have processed all the results
        #
        # The MA solves for a board size and also the mirror board size.
        # However the results are only stored in one of the folders.
        # For the results to be correct we need to update the extracted
        # solns so that it shows both board and mirror board results.
        #
        if self.algorithm[:2] == "MA":
            self.correct_solns()

    def correct_solns(self):
        """
        Some algorithms are able to solve for both the current board dimensions
        and the mirror board dimensions. This means they only produce one set
        of text results that are not picked up and must be corrected for.
        For the results to be correct we need to update the extracted
        solutions for both current and mirror board dimensions.
        Updates: self.solutions
        :return: n/a
        """
        # The board size is used as the key for each entry.
        # ie. {<mxn>:[<Python version>, <OS>, <execution time>, ...]}
        keys = self.solutions.keys()
        new_dictionary = False
        solns = self.solutions.copy()
        for key in keys:
            m, n = key.split("x")
            mirror_key = "{}x{}".format(n, m)
            if mirror_key in keys:
                # Solution already exists in dictionary of all solutions;
                # this means we don't need to add it.
                continue
            # Add the mirror solution
            new_dictionary = True
            soln_values = []
            print("Adding mirror key {}".format(mirror_key)) # debug
            for value in self.solutions[key]:
                soln_values.append(value)
            solns[mirror_key] = soln_values
        if new_dictionary:
            self.solutions = solns

    def create_test_coverage(self):
        """
        Creates a test results coverage text file for the specified algorithm,
        see name.
        self.solutions: (dict)solutions for algorithm 'name'.
        self.algorithm: (string)algorithm name.
        :param: n/a
        :return: n/a
        """
        print("\nTest coverage: processing results for the {} algorithm."
              .format(self.algorithm))
        # Extract all the keys, these represent the mxn board dimensions and
        # indicate what has been tested.
        keys = self.solutions.keys()
        # Extract the dimensions; use a regex
        regex_mxn = re.compile(r"(\d+)x(\d+)", re.IGNORECASE)
        m_max = int(0)
        n_max = int(0)
        mxn = []
        for keystr in keys:
            mxn_tuple = re.findall(regex_mxn, keystr)
            m_current = int(mxn_tuple[0][0])
            n_current = int(mxn_tuple[0][1])
            mxn.append((m_current, n_current))
            # Determine the max m and n ranges for our results table.
            if m_current > m_max:
                m_max = m_current
            if n_current > n_max:
                n_max = n_current
#        print("m_max={}, n_max={}".format(m_max, n_max))  # debug only
        # Create a 2D array and fill with zeros, then update with the tests found.
        m_max = m_max + 1   # Need to add an extra row for the board coordinates
        n_max = n_max + 1   # Need to add an extra column for the board coordinates
        table = [[' '] * n_max for _ in range(m_max)]
        # Add board coordinates to the table
        for idx in range(n_max):
            table[0][idx] = idx
        for idx in range(m_max):
            table[idx][0] = idx
        table[0][0] = "m/n"
        for coords in mxn:
            m_current = coords[0]
            n_current = coords[1]
#            print("{} x {}".format(m_current, n_current))   # debug only
            table[m_current][n_current] = 'X'
#        print(table)    # debug only
        # Now process the populated table and create a text file.
        ftext = "The following board dimensions have been tested.\n" \
                "Algorithm = {}\n".format(self.algorithm)
        tab = tt.Texttable()
        tab.add_rows(table)
        ftext += tab.draw()
        ftext += "\n\n\n"
        # Add the LaTeX table to the bottom of the file.
        table_title_str = "{} Test Coverage.".format(self.algorithm)
        ftext += self.get_latex_coverage(m_max, n_max, table, table_title_str)
#        print(ftext)   # debug only
        # Have all the data to write out to a results file.
        fname = os.path.join(cmn.RESULTS_FOLDER,
                             "{}_testcoverage.txt".format(self.algorithm))
        with open(fname, "w") as fout:
            fout.write(ftext)
        print("Created {}".format(fname))

    def create_colour_chart(self, kmax):
        """
        Creates a 2D colour chart of integer solutions vs board dimensions.
        Ref: https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/
        interpolate.html#two-dimensional-spline-representation-procedural-bisplrep
        self.solutions: (dict)solutions for algorithm 'name'.
        self.algorithm: (string)algorithm name.
        :param kmax: (int)maximum m or n dimension: m = n = (kmax - 1).
        :return: n/a
        """
        print("Results Graph: processing results for the {} algorithm."
              .format(self.algorithm))
        # Extract all the keys, these represent the mxn board dimensions and
        # indicate what has been tested.
        keys = self.solutions.keys()
        # Extract the dimensions; use a regex
        # We need these values to size the graph correctly
        regex_mxn = re.compile(r"(\d+)x(\d+)", re.IGNORECASE)
        # Now create a colour chart with the data
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        # Process each figure plot in turn: column by column and row by row
        for row in range(2):
            # Set the solution index to either all or fundamental solutions
            if row:
                idx = I_FUNSOLNS
            else:
                idx = I_ALLSOLNS
            for col in range(2):
                z_all = np.array([], np.uint32)
                points = np.array([], np.uint32)
                for keystr in keys:
                    mxn_tuple = re.findall(regex_mxn, keystr)
                    # Record all the points in arrays
                    # Populate the z_all array
                    m_current = int(mxn_tuple[0][0])
                    n_current = int(mxn_tuple[0][1])
                    z_all = np.append(z_all, self.solutions[keystr][idx])
                    points = np.append(points, [m_current, n_current])
                # Reshape the array into coordinates (m, n)
                points = np.reshape(points, (int(len(points) / 2), 2))
                # Reshape the results array z_all
                z_all = np.reshape(z_all, (int(len(z_all)), ))
                # np.linspace(start, stop, number of points)
                # eg. np.linspace(0, 100, 5) => 0, 25, 50, 75, 100
                if col == 1:
                    # High resolution grid - fascinating alternate view for colour map
                    grid_x, grid_y = np.meshgrid(np.linspace(1, kmax, kmax * 10),
                                                 np.linspace(1, kmax, kmax * 10))
                else:
                    # Low resolution grid
                    grid_x, grid_y = np.meshgrid(np.linspace(1, kmax, kmax),
                                                 np.linspace(1, kmax, kmax))
                grid_z = griddata(points, z_all, (grid_x, grid_y), method='cubic')

                # Add a single chart to the 2 x 2 figure
                ax[row, col].imshow(grid_z.T, extent=(0, kmax, 0, kmax), origin='lower')
                # Configure the major ticks; starting at -0.5 to ensure the
                # grid/board is drawn completely
                ax[row, col].set_xticks(list(range(kmax)), minor=False)
                ax[row, col].set_yticks(list(range(kmax)), minor=False)
                # Hide the labels for the major ticks on both axis.
                ax[row, col].tick_params(axis='both', which='major', labelcolor='white')
                # Configure the minor ticks; we want to use these to number the
                # rows and columns and centre them at the edge of each square.
                # Remember the rows and columns start at the bottom left corner
                # with at (0, 0).
                ax[row, col].xaxis.set_minor_locator(AutoMinorLocator(2))
                ax[row, col].yaxis.set_minor_locator(AutoMinorLocator(2))
                # Add the text labels to the minor ticks for both rows and columns.
                col_labels = [str(x + 1) for x in range(kmax)]
                ax[row, col].set_xticklabels(col_labels, minor=True)
                row_labels = [str(y + 1) for y in range(kmax)]
                ax[row, col].set_yticklabels(row_labels, minor=True)
                # Hide the ticks on both axis; looks better.
                ax[row, col].tick_params(axis='both', which='both', length=0.0)
                if idx == I_ALLSOLNS:
                    ax[row, col].set_title('All', fontsize=16)
                else:
                    ax[row, col].set_title('Fundamental', fontsize=16)
                ax[row, col].set_xlabel('m', fontsize=16)
                ax[row, col].set_ylabel('n', fontsize=16)
        # Set the graph size
        fig.suptitle("{} Solutions".format(self.algorithm),
                     fontsize=18, weight='bold')
        plt.subplots_adjust(hspace=0.3)
        # Have all the data to write out to a results file.
        fname = os.path.join(cmn.RESULTS_FOLDER,
                             "{}_solutions.png".format(self.algorithm))
        plt.savefig(fname)
#        plt.show() # debug
        print("Created {}".format(fname))

    def create_time_graph(self):
        """
        Creates a 2D colour chart of the time taken to find the
        solutions vs board dimensions. Where m = n only.
        Ref: https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/
        interpolate.html#two-dimensional-spline-representation-procedural-bisplrep
        self.solutions: (dict)solutions for algorithm 'name'.
        self.algorithm: (string)algorithm name.
        :param:  n/a
        :return: n/a
        """
        # Extract all times for when m = n
        print("\nTime graph: processing results for the {} algorithm."
              .format(self.algorithm))
        # Extract all the keys, these represent the mxn board dimensions and
        # indicate what has been tested.
        keys = self.solutions.keys()
        mxn_keys = []
        regex_mxn = re.compile(r"(\d+)x(\d+)", re.IGNORECASE)
        for keystr in keys:
            mxn_tuple = re.findall(regex_mxn, keystr)
            m_current = int(mxn_tuple[0][0])
            n_current = int(mxn_tuple[0][1])
            # If m = n then add to the list to create the graph.
            if m_current != n_current:
                continue
            mxn_keys.append(m_current)
        # Important: Need to sort the key values into order for the graph!
        keys = sorted(mxn_keys)
        # Extract the dimensions; use a regex
        graph_labels = []
        graph_tsecs = []
        graph_fsecs = []
#        print(keys) # debug only
        for key in keys:
            keystr = "{0}x{0}".format(key)
            graph_labels.append(key)
            graph_tsecs.append(self.solutions[keystr][I_EXETIME])
            graph_fsecs.append(self.solutions[keystr][I_1STTIME])

#        print(graph_labels) # debug only
#        print(graph_tsecs)  # debug only
#        print(graph_fsecs)  # debug only
        # Now create a colour chart with the data
        plt.figure()
        plt.subplot(211)
        plt.plot(graph_labels, graph_fsecs, 'r') # time to first solution
        plt.title("{}:Time to first solution".format(self.algorithm))
        plt.ylabel("Time (s)")

        plt.subplot(212)
        plt.plot(graph_labels, graph_tsecs, 'b') # time for all solutions
        plt.title("{}: Time to all solutions".format(self.algorithm))
        plt.xlabel("Board Dimensions (m = n)")
        plt.ylabel("Time (s)")

        plt.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
        # Suggested defaults and their meaning.
        # left = 0.125  # the left side of the subplots of the figure
        # right = 0.9   # the right side of the subplots of the figure
        # bottom = 0.1  # the bottom of the subplots of the figure
        # top = 0.9     # the top of the subplots of the figure
        # wspace = 0.2  # the amount of width reserved for blank space between subplots
        # hspace = 0.2  # the amount of height reserved for white space between subplots

        # Have all the data to write out to a results file.
        fname = os.path.join(cmn.RESULTS_FOLDER,
                             "{}_time_solutions.png".format(self.algorithm))
        plt.savefig(fname)
#        plt.show() # debug
        print("Created {}".format(fname))

    def extract_integer_sequences(self):
        """
        Extracts the Integer sequence for fundamental and all solutions.
        For all m x n board dimensions.
        The results are written to a text file.
        self.solutions: (dict)solutions for algorithm 'name'.
        self.algorithm: (string)algorithm name, eg GA, BA, MA.
        :param: n/a
        :return: n/a
        """
        print("\nInteger Sequence: processing results for the {} algorithm."
              .format(self.algorithm))
        # Extract all the keys, these represent the mxn board dimensions and
        # indicate what has been tested.
        # Note: keys is a view, NOT a list.
#        print(list(self.solutions)[0])
        keys = self.solutions.keys()
        # Extract the dimensions; use a regex
        regex_mxn = re.compile(r"(\d+)x(\d+)", re.IGNORECASE)
        m_max = int(0)
        n_max = int(0)
        mxn = []
        #
        # Run through all the results and determine the results table dimensions
        #
        for keystr in keys:
            mxn_tuple = re.findall(regex_mxn, keystr)
            m_current = int(mxn_tuple[0][0])
            n_current = int(mxn_tuple[0][1])
            mxn.append((m_current, n_current))
            # Determine the max m and n ranges for our results table.
            if m_current > m_max:
                m_max = m_current
            if n_current > n_max:
                n_max = n_current
#        print("m_max={}, n_max={}".format(m_max, n_max))  # debug only
        #
        # Create a 2D list array and fill with spaces, then update with the tests found.
        #
        m_max = m_max + 1   # Need to add an extra row for the board coordinates
        n_max = n_max + 1   # Need to add an extra column for the board coordinates
        table = [[' '] * n_max for _ in range(m_max)]
        # Add board coordinates to the table
        for idx in range(n_max):
            table[0][idx] = idx
        for idx in range(m_max):
            table[idx][0] = idx
        table[0][0] = "m/n"
        for coords in mxn:
            m_current = coords[0]
            n_current = coords[1]
#            print("{} x {}".format(m, n))   # debug only
            # I_ALLSOLNS = 4
            # I_FUNSOLNS = 5
            # Load the table with the an integer string of '<fundamental>-<all>' solutions.
            key = "{}x{}".format(m_current, n_current)
            iseq_str = "{}-\n{}".format(self.solutions[key][I_FUNSOLNS], self.solutions[key][I_ALLSOLNS])
            table[m_current][n_current] = iseq_str
#        print(table)    # debug only
        # Now process the populated table and create a text file.
        ftext = "The following board dimensions have been tested.\n" \
                "Algorithm = {}\n".format(self.algorithm)
        tab = tt.Texttable()
        tab.add_rows(table)
        # Set the table column width, first column is thinner as this is just the coordinate.
        col_width = [8 for _ in range(n_max - 1)]
        col_width.insert(0, 3)
        tab.set_cols_width(col_width)
        # Now ready to draw the table.
        ftext += tab.draw()
        ftext += "\n\nFormat: <fundamental>-<all> Integer solutions\n"
        ftext += "\n\n\n"
        # Add the LaTeX table to the bottom of the file.
        table_title_str = "{} Integer Sequence.".format(self.algorithm)
        ftext += self.get_latex_sequence(m_max, n_max, table, table_title_str)
#        print(ftext)   # debug only
        # Have all the data to write out to a results file.
        fname = os.path.join(cmn.RESULTS_FOLDER, "{}_iSequences.txt"
                             .format(self.algorithm))
        with open(fname, "w") as fout:
            fout.write(ftext)
        print("Created {}".format(fname))

    def get_latex_coverage(self, rows, cols, tarray, title):
        """
        Creates a latex table as a string.
        The string can be cut and paste into the dissertation.
        :param rows:   number of rows in the board.
        :param cols:   number of columns in the board.
        :param tarray: 2D array of the table contents.
        :param title:  Table string title.
        :return: a string of the completed LaTeX Table.
        """
        latex_str = "\\begin{table}[ht]\n"
        latex_str += "\\centering\n"
        latex_str += "\\begin{small}\n"
        latex_str += "\\setlength{\\arrayrulewidth}{0.5pt}"
        latex_str += " % thickness of the borders of the table\n"
        latex_str += "\\setlength{\\tabcolsep}{1.0em}"
        latex_str += " %  space between the text and the left/right border\n"
        latex_str += "\\renewcommand{\\arraystretch}{1.25}"
        latex_str += " % height of each row\n"
        latex_str += "\\arrayrulecolor{black}\n"
        latex_str += "\\begin{tabular}{|p{1.6em}|"
        # Configure the number of columns
        for _ in range(cols - 1):
            latex_str += "p{0.6em}|"
        latex_str += "}\n"
        # We are now into the table contents!
        for row in range(rows):
            latex_str += "\\hline\n"
            for col in range(cols):
                cell_contents = "{}".format(tarray[row][col])
                # Ensure the table coorindate cells are in bold font
                if col == row:
                    cell_contents = "\\cellcolor{blue!25}\\textbf{" \
                                    + cell_contents + "}"
                elif col == 0 or row == 0:
                    cell_contents = "\\cellcolor{gray!60}\\textbf{" \
                                    + cell_contents + "}"
                # Ensure the all diagonal cells are highlighted
                latex_str += " " + cell_contents + " "
                if (col + 1) == cols:
                    latex_str += "\\\\\n"
                else:
                    latex_str += "&"
        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\end{small}\n"
        latex_str += "\\caption{" + title + "}\n"
        latex_str += "\\label{tab:" + title.replace(" ", "") + "}\n"
        latex_str += "\\end{table}\n"
        return latex_str

    def get_latex_sequence(self, rows, cols, tarray, title):
        """
        Creates a latex table as a string.
        The string can be cut and paste into the dissertation.
        :param rows:   number of rows in the board.
        :param cols:   number of columns in the board.
        :param tarray: 2D array of the table contents.
        :param title:  Table string title.
        :return: a string of the completed LaTeX Table.
        """
        latex_str = "\\begin{sidewaystable}\n"
        latex_str += "\\centering\n"
        latex_str += "\\begin{tiny}\n"
        latex_str += "\\setlength{\\arrayrulewidth}{0.5pt}"
        latex_str += " % thickness of the borders of the table\n"
        latex_str += "\\setlength{\\tabcolsep}{2.5em}"
        latex_str += " %  space between the text and the left/right border\n"
        latex_str += "%\\renewcommand{\\arraystretch}{2.5em}"
        latex_str += " % height of each row\n"
        latex_str += "\\arrayrulecolor{black}\n"
        latex_str += "\\begin{tabular}{|"
        # Configure the number of columns
        for _ in range(cols):
            latex_str += "p{0.5em}|"
        latex_str += "}\n"
        # We are now into the table contents!
        for row in range(rows):
            latex_str += "\\hline\n"
            for col in range(cols):
                cell_contents = "{}".format(tarray[row][col]).replace("\n", "")
                # Ensure the table coorindate cells are in bold font
                if col == row:
                    cell_contents = "\\cellcolor{blue!25}\\textbf{" \
                                    + cell_contents + "}"
                elif col == 0 or row == 0:
                    cell_contents = "\\cellcolor{gray!60}\\textbf{" \
                                    + cell_contents + "}"
                # Ensure the all diagonal cells are highlighted
                latex_str += " " + cell_contents + " "
                if (col + 1) == cols:
                    latex_str += "\\\\\n"
                else:
                    latex_str += "&"
        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\end{tiny}\n"
        latex_str += "\\caption{" + title + "}\n"
        latex_str += "\\label{tab:" + title.replace(" ", "") + "}\n"
        latex_str += "\\end{sidewaystable}\n"
        return latex_str


def get_latex_permutations(rows, cols, tarray, title):
    """
    Creates a latex table as a string.
    The string can be cut and paste into the dissertation.
    :param rows:   number of rows in the board.
    :param cols:   number of columns in the board.
    :param tarray: 2D array of the table contents.
    :param title:  Table string title.
    :return: a string of the completed LaTeX Table.
    """
    latex_str = "\\begin{sidewaystable}\n"
    latex_str += "\\centering\n"
    latex_str += "\\begin{small}\n"
    latex_str += "\\setlength{\\arrayrulewidth}{0.5pt}"
    latex_str += " % thickness of the borders of the table\n"
    latex_str += "\\setlength{\\tabcolsep}{0.75em}"
    latex_str += " %  space between the text and the left/right border\n"
    latex_str += "\\arrayrulecolor{black}\n"
    latex_str += "\\begin{tabular}{|"
    # Configure the number of columns
    for _ in range(cols):
        latex_str += "c|"
    latex_str += "}\n"
    # We are now into the table contents!
    for row in range(rows):
        latex_str += "\\hline\n"
        for col in range(cols):
            cell_contents = "{}".format(tarray[row][col]).replace("\n", "")
            # Ensure the table coorindate cells are in bold font
            if col == row:
                cell_contents = "\\cellcolor{blue!25}\\textbf{" \
                                + cell_contents + "}"
            elif col == 0 or row == 0:
                cell_contents = "\\cellcolor{gray!60}\\textbf{" \
                                + cell_contents + "}"
            # Ensure the all diagonal cells are highlighted
            latex_str += " " + cell_contents + " "
            if (col + 1) == cols:
                latex_str += "\\\\\n"
            else:
                latex_str += "&"
    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{small}\n"
    latex_str += "\\caption{" + title + "}\n"
    latex_str += "\\label{tab:" + title.replace(" ", "") + "}\n"
    latex_str += "\\end{sidewaystable}\n"
    return latex_str

def all_permutations():
    """
    Extract the results of calculating all the permutations for the
    specified board dimensions and create a table.
    :param:  n/a
    :return: n/a
    """
    # Create a list of permutations solutions files
    # Ensure the results folder actually exists.
    folder = cmn.PERMUTATIONS_FOLDER
    if os.path.exists(folder):
        # The results folder exists; now verify that it contains any results
        # data. Scan the folder for files containing actual results.
        permutations = [name for name in os.listdir(folder) \
                        if os.path.isfile(os.path.join(folder, name))]
    else:
        permutations = []
    # Create a dictionary of solutions and mirror results
    solns = {}
    regex_val = re.compile(r"Number of permutations\s*=\s*(\d+)", re.MULTILINE)
    for fname in permutations:
        print("Processing file {}".format(fname))
        # Read in the entire results text file
        with open(os.path.join(folder, fname), "r") as fin:
            contents = fin.read()
        # Now extract the permutations for this board size.
        value = int(re.findall(regex_val, contents)[0])
        # Create the dictionary key from the board dimensions.
        key = fname.split("-")[0]
        if key not in solns:
            solns[key] = value
        # Mirror the board dimensions
        m, n = key.split("x")
        key = "{}x{}".format(n, m)
        if key not in solns:
            solns[key] = value
#    print(solns) # debug

    #
    # Create a 2D list array and fill with spaces, then update with the tests found.
    #
    m_max = n_max = 13 + 1 # extra row and column for the board coordinates
    table = [[' '] * n_max for _ in range(m_max)]
    # Add board coordinates to the table
    for idx in range(n_max):
        table[0][idx] = idx
    for idx in range(m_max):
        table[idx][0] = idx
    table[0][0] = "m/n"
    for key in solns:
        m_current = int(key.split("x")[0])
        n_current = int(key.split("x")[1])
        # Load the table with the permutations integer.
        table[m_current][n_current] = solns[key]
#        print(table)    # debug only
    # Now process the populated table and create a text file.
    ftext = "The following board dimensions have been tested to calculate the"
    ftext += " total number of permutations.\n"
    tab = tt.Texttable()
    tab.add_rows(table)
    # Set the table column width, first column is thinner as this is just the coordinate.
    col_width = [8 for _ in range(n_max - 1)]
    col_width.insert(0, 3)
    tab.set_cols_width(col_width)
    # Now ready to draw the table.
    ftext += tab.draw()
    ftext += "\n\n\n"
    # Add the LaTeX table to the bottom of the file.
    table_title_str = "Total Permutations."
    ftext += get_latex_permutations(m_max, n_max, table, table_title_str)
#    print(ftext)   # debug only
    # Have all the data to write out to the permutations results file.
    fname = cmn.PERMUTATIONS_FILE
    with open(fname, "w") as fout:
        fout.write(ftext)
    print("Created {}".format(fname))


def main():
    """
    Main results script function to parse the BA, GA and MA results.
    :param: None
    :return: n/a
    """
    #
    # Process the specified results folder
    #
    for folder in cmn.RESULTS_FOLDERS:
        print("\n" + "-" * 50)
        print("Processing {}".format(folder))
        # Create the object and extract all the results folders
        results = CResults(folder)
        # No point continuing if there are not results
        if results.empty():
            print("No results found.\n")
            continue
        #
        # Extract: board dimensions, execution time, total solutions and fundamental solutions.
        #
        print("Extracting all solutions from the {} results folder."
              .format(results.algorithm))
        results.extract_solns()
        #
        # Create text file: display m x n board and results (test coverage)
        #
        print("Extracting information to determine the test coverage.")
        results.create_test_coverage()
        #
        # Create 2D colour chart of solutions
        #
        results.create_colour_chart(15)
        #
        # Create Integer Sequences
        #
        results.extract_integer_sequences()
        #
        # Create graph of time to find solutions.
        #
        results.create_time_graph()
    #
    # Extract the results of calculating all the permutations for the specified
    # board dimensions and create a table.
    #
    all_permutations()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--version', action='version', version=SCRIPTINFO)
    # Make sure we process any command line arguments.
    vars(PARSER.parse_args())

    # Start up the application
    main()

# EOF

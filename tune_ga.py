#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
Script to help identify and tune the GA to the optimal values for: population
size, mutation rate, cross-over rate and maximum number of generations. There
must be a correlation between between all these variables and with the board
dimensions m x n.

Command line usage example:
    py -3 tune_ga.py
"""

import os
import sys
import argparse

import importlib
import shutil
import matplotlib.pyplot as plt
import common as cmn
import ga

__author__ = 'David Kind'
__date__ = '30-01-2020'
__version__ = '1.2'
__copyright__ = 'Copyright(c) 2019 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)

DEFAULTS_TEST = "Defaults"
MUTATION_RATE_TEST = "Mutation rate"
POPULATION_TEST = "Population size"
CROSSOVER_TEST = "Cross-over rate"
GENERATIONS_TEST = "Maximum generations"

# Note: only testing board sizes for m = n.
BOARDS = [['2', '2'], ['3', '3'], ['4', '4'], ['5', '5'], ['6', '6'], ['7', '7'], ['8', '8'], ['9', '9']]
TEST_RUNS = int(20)  # Number of runs to perform


class CTest(object):
    """
    Test Results Object.
    """
    # Working variables
    name = ""                       # Name of the test
    GENERATIONS_MAX = int(0)        # GA maximum number of generations
    POPULATION_SIZE = int(0)        # GA population size
    CX_PROBABILITY = float(0.0)     # GA cross-over probability
    MUTATION_RATE = float(0.0)      # GA mutation rate
    solution_times = []             # GA time to find a solution for each board
    solution_number = []            # GA number of solutions for each board

    def __init__(self, name, generations, population, cx_probability, mutation_rate):
        # Initialise the object members
        self.name = name
        # Set the GA values or defaults if None
        if generations:
            ga.GENERATIONS_MAX = generations
        self.GENERATIONS_MAX = ga.GENERATIONS_MAX
        if population:
            ga.POPULATION_SIZE = population
        self.POPULATION_SIZE = ga.POPULATION_SIZE
        if cx_probability:
            ga.CX_PROBABILITY = cx_probability
        self.CX_PROBABILITY = ga.CX_PROBABILITY
        if mutation_rate:
            ga.MUTATION_RATE = mutation_rate
        self.MUTATION_RATE = ga.MUTATION_RATE
        # Initialise the solution lists for the board sizes
        # These are average values so need to be floats.
        self.solution_times = [float("inf") for _ in range(len(BOARDS))]
        self.solution_number = [float(0) for _ in range(len(BOARDS))]

    def reset_params(self):
        """
        Reset the GA parameters as these are reset each time the GA is reloaded.
        :return: n/a
        """
        ga.GENERATIONS_MAX = self.GENERATIONS_MAX
        ga.POPULATION_SIZE = self.POPULATION_SIZE
        ga.CX_PROBABILITY = self.CX_PROBABILITY
        ga.MUTATION_RATE = self.MUTATION_RATE

    def plot(self):
        """
        Create a plot of a the solutions using a graph with 2x y-axes.
        :return: n/a
        """
        _, ax1 = plt.subplots()
        x = [val for val in range(2, len(BOARDS) + 2)]
        ax2 = ax1.twinx()
        ax1.plot(x, self.solution_times, 'g-')
        ax2.plot(x, self.solution_number, 'b-')
        ax1.set_xlabel('Board size m = n')
        ax1.set_ylabel('Average solution time (s)', color='g')
        ax2.set_ylabel('Average number of solutions', color='b')
        # Ready to add the title to the result plot
        soln_title = "{}: gen={}, pop={}, cx={:.2f}, mut={:.2f}" \
            .format(self.name, ga.GENERATIONS_MAX, ga.POPULATION_SIZE, ga.CX_PROBABILITY, ga.MUTATION_RATE)
        plt.title(soln_title)
        # Now show/save the resultant plot.
        soln_path = os.path.join(cmn.RESULTS_FOLDER, "GA_tuning")
        # Make sure to create the directory if not already there.
        if not os.path.exists(soln_path):
            os.makedirs(soln_path)
        # fname format: test_<name>-<generations>-<population>-<cross-over>-<mutation>
        fname = "test_{}-{}-{}-{:.2f}-{:.2f}.png" \
            .format(self.name, ga.GENERATIONS_MAX, ga.POPULATION_SIZE, ga.CX_PROBABILITY, ga.MUTATION_RATE)
        soln_path = os.path.join(soln_path, fname)
        plt.savefig(soln_path)
        print("\nPlotting results to {}\n".format(soln_path))
#        plt.show()     # Instant displayed plot
        # Be sure to clear down the plt object, otherwise is clogs up and
        # eventually runs out of memory.
        plt.close('all')

    def __str__(self):
        # Object string: used for debug purposes
        outstr = "Test name:               {}\n".format(self.name)
        outstr += "Generations maximum    = {}\n".format(self.GENERATIONS_MAX)
        outstr += "Population size        = {}\n".format(self.POPULATION_SIZE)
        outstr += "Cross-over probability = {}\n".format(self.CX_PROBABILITY)
        outstr += "Mutation rate          = {}\n".format(self.MUTATION_RATE)
        outstr += "Solution times:          {}\n".format(self.solution_times)
        outstr += "Solutions found:         {}\n".format(self.solution_number)
        return outstr


def main(name, gen, pop, cx, mut):
    """
    Characterising the following variables:
        1) m x n
        2) mutation rate
        3) population size
        4) cross-over rate
        5) maximum generations
    Evaluating GA performance:
        1) Number of solutions
        2) Time to solve
    :param name: String of the test name.
    :param gen:  maximum number of GA generations or None for default.
    :param pop:  population size or None for default.
    :param cx:   cross-over rate or None for default.
    :param mut:  mutation rate or None for default.
    :return: n/a
    """
    # Remove all previous results from the results folder
    path_str = os.path.join(cmn.RESULTS_FOLDER, "GA")
    if os.path.isdir(path_str):
        print("*************************************************")
        print("Removing all previous results from the GA folder.")
        print("*************************************************\n")
        shutil.rmtree(path_str)

    # The default values represent the base control group
    print("Running the tests for {}.".format(name))
    test = CTest(name, gen, pop, cx, mut)
    for idx, mxn in enumerate(BOARDS):
        total_solns = float(0)
        total_time = float(0)
        for _ in range(TEST_RUNS):
            # Execute the Genetic Algorithm for the specified board size
            test.reset_params()
            exe_time, solns = ga.main(mxn, False)
            total_solns = total_solns + solns
            if solns:
                total_time = total_time + (exe_time / solns)
            else:
                total_time = total_time + exe_time
            # Clean up the objects, so we can run the test again
            del ga.creator.FitnessMax
            del ga.creator.Individual
            importlib.reload(ga)
        if total_solns:
            test.solution_number[idx] = float(total_solns / TEST_RUNS)
            test.solution_times[idx] = float(total_time / TEST_RUNS)
        else:
            test.solution_times[idx] = float("inf")
    # Dump out the raw results of the testing
    test.reset_params()
    print("\n{}\n".format(test))
    # Plot the results
    test.plot()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--version', action='version', version=SCRIPTINFO)
    PARSER.add_argument('--generations', '-g',
                        action='store',
                        nargs='?',
                        type=int,
                        help='Maximum number of generations (GA cost).')
    PARSER.add_argument('--population', '-p',
                        action='store',
                        nargs='?',
                        type=int,
                        help='GA population size.')
    PARSER.add_argument('--crossover', '-c',
                        action='store',
                        nargs='?',
                        type=float,
                        help='GA cross-over rate [0, 1).')
    PARSER.add_argument('--mutation', '-m',
                        action='store',
                        nargs='?',
                        type=float,
                        help='GA mutation rate [0, 1).')
    PARSER.add_argument('name', nargs=1,
                        help='String name for test.')
    # Get the arguments dictionary, where arguments are the keys.
    ARGS = vars(PARSER.parse_args())
    # Extract the grid dimensions as a list [x, y]
    NAME = ARGS['name']
    GENERATIONS = ARGS['generations']
    POPULATION = ARGS['population']
    CX = ARGS['crossover']
    MUTATION = ARGS['mutation']
#    print("NAME={}, GENERATIONS={}, POPULATION={}, CX={}, MUTATION={}"
#          .format(NAME[0], GENERATIONS, POPULATION, CX, MUTATION))    # debug

    # Start up the application
    print("Running: {}".format(SCRIPTINFO))
    main(NAME[0], GENERATIONS, POPULATION, CX, MUTATION)

# EOF

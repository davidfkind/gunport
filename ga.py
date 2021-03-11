#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
Genetic Algorithm to solve The Gun Port Problem as first described by
Sands, B. (1971), The Gunport Problem, Mathematics Magazine, Vol.44, pp.193-196

This is an optimisation problem to maximise the holes and minimise the number
number of dominoes in a board of variable size. Holes are not allowed to touch
each other side on.

Command line usage example:
    py -3 ga.py 6x6
"""

import os
import sys
import argparse
import random
import time                     # Used to time script execution.
import numpy as np

from deap import base
from deap import creator
from deap import tools

import domplotlib as dpl        # Domino plotting library
import common as cmn            # Common defines and functions

__author__ = 'David Kind'
__date__ = '21-01-2020'
__version__ = '3.1'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)

#
# GA Parameters that can be modified to alter performance
# Note: mutation rate affects the convergence speed.
#
GENERATIONS_MAX = int(2000)     # The total number of generations
POPULATION_SIZE = int(1000)     # The total number of individuals; must be even.
CX_PROBABILITY = float(0.0)     # (%) probability with which two individuals
                                # are crossed, range [0.0, 1.0)
MUTATION_RATE = float(0.25)     # (%) probability of mutating an individual,
                                # range [0.0, 1.0)

class CEvaluation(object):
    """
    GA Evaluation Object.
    """
    chromosome = []     # copy of the individual
    weight = float(0)   # (%) individual weight/fitness rating
    board = None        # Resultant board with dominoes placed
    spaces = int(0)     # Number of empty spaces in the fitted board
    dominoes = int(0)   # Number of dominoes placed in the fitted board
    max_spaces = int(0) # Maximum number of calculated spaces
    max_dominoes = int(0) # Maximum number of calculated dominoes
    xsize = int(0)      # Board width
    ysize = int(0)      # Board height
    biased = True       # Biased selection favouring holes over dominoes
    toggle = False      # Toggle between domino types

    def __init__(self, x_width, y_height, max_spaces, max_dominoes):
        # Initialise the object members
        self.spaces = int(0)
        self.dominoes = int(0)
        self.max_spaces = max_spaces
        self.max_dominoes = max_dominoes
        self.xsize = x_width
        self.ysize = y_height
        # Set an initial weighting.
        self.weight = float(0)
        self.biased = True
        self.toggle = False

    def eval_fitness(self, individual):
        """
        Fits the chromosome to a blank board correcting any issues so that the
        resulting pattern is acceptable; this could mean replacing/removing or
        adding some of the shapes. The resultant board contains only 0s and
        values > 1, where values > 1 represent a domino and 0 represents a hole.
        Check for:
          (1) empty spaces next to each other, testing above and left only.
          (2) replacing overlapping dominoes.
          (3) removing unused genes from the end of the list.
        :param individual: DEAP individual to be evaluated.
        """
        self.chromosome = individual
#        print("orig: {}".format(individual)) # debug
        self.board = np.zeros((self.ysize, self.xsize))
        self.spaces = int(0)    # Running total of spaces in board
        self.dominoes = int(0)  # Running total of dominoes in board
        idx = 0                 # Chromosome index
        for row in range(self.ysize):
            for col in range(self.xsize):
                #
                # Is the board location already occupied
                #
                if self.board[row, col] != cmn.CELL_SPACE:
                    # It is occupied move along to the next position
                    continue
                # Create a list of shapes that can be placed at this location.
                # There are 3x types: hole, vertical domino and horizontal
                # domino. If present then the shape can be fitted to the
                # specified location in the board.
                ok_shapes = self.check_board(row, col)
#                print("({}x{}): current shape [{}]={}; ok_shapes={}"
#                      .format(row, col, idx, individual[idx], ok_shapes)) # debug
                #
                # Set the contents of the current board location using the
                # stored value if possible. Get a list of valid shapes and
                # verify if the current shape is valid, if not select from
                # the list of current valid shapes.
                #
                num_ok_shapes = len(ok_shapes)
                if individual[idx] in ok_shapes:
#                    print("=> current shape ok") # debug
                    self.set_board(row, col, individual[idx])
                # The current value (shape) does not fit.
                # Need to try and fit a shape to this location and other shapes
                # are able to fit in this board location.
                elif num_ok_shapes:
                    # Need to randomly select a shape to place in this location
                    # from the shapes that are available.
                    val = ok_shapes[random.randint(0, num_ok_shapes - 1)]
                    self.set_board(row, col, val)
                    individual[idx] = val
#                    print("=> new shape selected {}".format(individual[idx])) # debug
                # Nothing is OK, but we have a space either above or to
                # the left so can replace that with a domino. This is
                # in all likelihood the last board square.
                else:
                    # Adjust the spaces and domino count as we're going to have
                    # replace a space with a domino.
                    self.spaces -= 1
                    self.dominoes += 1
                    if col > 0 and self.board[row, col - 1] == cmn.CELL_SPACE:
                        # Replace space with Horizontal domino
                        individual[idx - 1] = cmn.CELL_HDOMINO
                        self.board[row, col - 1] = cmn.CELL_HDOMINO
                        self.board[row, col] = cmn.CELL_HDOMINO
                    else:
                        # Replace space with Vertical domino
                        if row > 0 and self.board[row - 1, col] == cmn.CELL_SPACE:
                            previdx = self.get_idx(individual, row - 1, col)
                            individual[previdx] = cmn.CELL_VDOMINO
                            self.board[row - 1, col] = cmn.CELL_VDOMINO
                            self.board[row, col] = cmn.CELL_VDOMINO
                        else:
                            print("ERROR: UNABLE TO PLACE SHAPE!!!")
#                    print("=> nothing ok") # debug
                    # Don't increase the idx counter as we may to re-evaluate
                    # the shape currently being pointed at.
                    continue
                # Look at the next individual element
                idx += 1
        # The individual could have been modified so we need to update our copy.
        self.chromosome = individual
        # Calculate the weighting out of 100.
        if self.spaces >= self.max_spaces:
            self.weight = 100
        else:
            self.weight = self.spaces * (100 / self.max_spaces)
        if self.toggle:
            self.toggle = False
        else:
            self.toggle = True
#        print("final: weight={}; {}".format(self.weight, individual)) # debug
#        print("-" * 80)                       # debug
        return self.weight,

    def get_idx(self, tind, trow, tcol):
        """
        Returns the chromosome index associated with the target coordinates.
        This is tricky because the different shapes mean the chromosome index
        does not relate to the board coordinates and we must recalculate to
        identify the required location.
        :param tind: board target chromosome.
        :param trow: board target row position.
        :param tcol: board target column position.
        :return: chromosome target integer index.
        """
        idx = 0
        row, col = 0, 0
        board = np.zeros((self.ysize, self.xsize))
        for idx in range(len(tind)):
            # Have we reached the bounds of the board?
            if row >= self.ysize:
                idx = None
                break
            # Find the next free board location
            while board[row, col] != cmn.CELL_SPACE:
                col = col + 1
                if col == self.xsize:
                    row = row + 1
                    col = 0
            if row == trow and col == tcol:
                break   # Found target coordinates
            if tind[idx] == cmn.CELL_SPACE:
                board[row, col] = cmn.CELL_SPACE
                col = col + 1
            elif tind[idx] == cmn.CELL_HDOMINO:
                board[row, col] = cmn.CELL_HDOMINO
                board[row, col + 1] = cmn.CELL_HDOMINO
                col = col + 2
            else: # tind[idx] == cmn.CELL_VDOMINO:
                board[row, col] = cmn.CELL_VDOMINO
                board[row + 1, col] = cmn.CELL_VDOMINO
                col = col + 1
            # Check to see if we have reached the right hand side of the board.
            if col == self.xsize:
                row = row + 1
                col = 0
        return idx

    def check_board(self, row, col):
        """
        Returns a list of valid shapes after verifying whether the shape can be
        fitted to the specified location in the board. Only checks the current
        location against proceeding locations and not the next location.
        Shapes are: CELL_SPACE, CELL_HDOMINO, CELL_VDOMINO
        Note: biased towards placing spaces/holes ahead of dominoes.
        :param row: board row position.
        :param col: board column position.
        """
        result = []
        # Are we ok to place have an empty space?
        if not ((col > 0 and self.board[row, col - 1] == cmn.CELL_SPACE) or
                (row > 0 and self.board[row - 1, col] == cmn.CELL_SPACE)):
            result.append(cmn.CELL_SPACE)
            # Favour holes over dominoes
            if self.biased:
                result.append(cmn.CELL_SPACE)
                result.append(cmn.CELL_SPACE)
                result.append(cmn.CELL_SPACE)
                result.append(cmn.CELL_SPACE)
                result.append(cmn.CELL_SPACE)
                result.append(cmn.CELL_SPACE)
                result.append(cmn.CELL_SPACE)
                result.append(cmn.CELL_SPACE)
        # Are we ok to have a horizontal domino?
        # Make sure we don't place the horizontal domino over previously placed
        # vertical domino on the board.
        if col < self.xsize - 1 and \
           self.board[row, col + 1] == cmn.CELL_SPACE:
            result.append(cmn.CELL_HDOMINO)
            # Favour horizontal dominoes
            if self.biased and not self.toggle:
                result.append(cmn.CELL_HDOMINO)
                result.append(cmn.CELL_HDOMINO)
                result.append(cmn.CELL_HDOMINO)
                result.append(cmn.CELL_HDOMINO)
        # Are we ok to have a vertical domino?
        if row < self.ysize - 1:
            result.append(cmn.CELL_VDOMINO)
            # Favour vertical dominoes
            if self.biased and self.toggle:
                result.append(cmn.CELL_VDOMINO)
                result.append(cmn.CELL_VDOMINO)
                result.append(cmn.CELL_VDOMINO)
                result.append(cmn.CELL_VDOMINO)
#        print("check_board: bias={} ({}x{}): xsize={}, ysize={}"
#              .format(self.biased, row, col, self.xsize, self.ysize))   # debug
#        print("result={}".format(result)) # debug
        return result

    def set_board(self, row, col, value):
        """
        Set the contents of the current board location using the value.
        Assumes the board has been zero'd before use.
        :param row: board row position.
        :param col: board column position.
        :param value: represents hole or domino to be placed at the specified location.
        """
        if value == cmn.CELL_SPACE:
            # Empty space is OK
            self.board[row, col] = cmn.CELL_SPACE
            self.spaces += 1
        elif value == cmn.CELL_HDOMINO:
            # Horizontal domino OK
            self.board[row, col] = cmn.CELL_HDOMINO
            self.board[row, col + 1] = cmn.CELL_HDOMINO
            self.dominoes += 1
        else:
            # value == cmn.CELL_VDOMINO:
            # Vertical domino OK
            self.board[row, col] = cmn.CELL_VDOMINO
            self.board[row + 1, col] = cmn.CELL_VDOMINO
            self.dominoes += 1

    def __str__(self):
        # Object string: used for debug purposes
        outstr = "\n{}\nfitness={}%".format(self.chromosome, self.weight)
        outstr += ", spaces={}".format(self.spaces)
        outstr += ", dominoes={}".format(self.dominoes)
        outstr += "\n{}".format(self.board)
        return outstr

def init_DEAP_structures(chromosome_size, cols, rows, max_holes, max_dominoes):
    """
    Initialises the DEAP framework for GA and MA use.
    :param chromosome_size: the maximum size of the chromosome.
    :param cols: number of columns for the specified board size.
    :param rows: number of rows for the specified board size.
    :param max_holes: solution maximum number of holes.
    :param max_dominoes: solution minimum number of dominoes.
    :return: the DEAP toolbox object, the DEAP population and fitness of each
             individual in the population.
    """
    # Initialise the random functionality
    random.seed(time.time())   # MUST ENSURE RANDOM SEED EACH RUN
    #
    # Set up the DEAP Framework...
    # Based on the examples provided in the DEAP documentation.
    # Ref: https://deap.readthedocs.io/en/master/examples/ga_onemax.html
    #
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Define the gene attribute, which correspond to 0=hole, 1=horizontal domino
    # and 2=vertical domino. # The definitions can be found in common.py.
    # The selection of the 3 types is performed with equal probability.
    # random.randint(a, b) => Returns a random integer N such that: a <= N <= b.
    toolbox.register("attr_int", random.randint, 0, 2)
    # Structure initializers.
    # - define 'individual' to be an individual
    # - consisting of chromosome_size 'attr_int' elements ('genes')
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, chromosome_size)
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #
    # Operator registration
    # Ref: https://deap.readthedocs.io/en/master/api/tools.html?highlight=cxtwopoint#operators
    #
    # register the goal / fitness function
    evalobj = CEvaluation(cols, rows, max_holes, max_dominoes)
    toolbox.register("evaluate", evalobj.eval_fitness)
    # register the crossover operator
    # Ref: https://deap.readthedocs.io/en/master/api/tools.html?highlight=cxtwopoint#deap.tools.cxPartialyMatched
    toolbox.register("mate", tools.cxPartialyMatched)
    # Register a mutation operator with a probability to change a chromosome
    # (list) entry to a value: 0 <= value <= 2
    # Ref: https://deap.readthedocs.io/en/master/api/tools.html?highlight=cxtwopoint#deap.tools.mutUniformInt
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.05)
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    # Ref: https://deap.readthedocs.io/en/master/api/tools.html?highlight=cxtwopoint#deap.tools.selTournament
#    toolbox.register("select", tools.selTournament)
#    toolbox.register("select", tools.selRoulette)
    toolbox.register("select", tools.selStochasticUniversalSampling)
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=POPULATION_SIZE)
#    print("Start of evolution")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("\nEvaluating {} individuals".format(len(pop)))
    # Extracting all the fitness values into a list
    fits = [ind.fitness.values[0] for ind in pop]
    evalobj.biased = False  # All shapes have equal probability
    return toolbox, pop, fits

def main(grid, timed_execution):
    """
    Main Genetic Algorithm implementation.
    :param grid: the board dimensions in rows and columns.
    :param timed_execution: flag to time algorithm execution.
    :return: execution time and total number of solutions found.
    """
    # Setup the GA configuration and show the user
    rows = int(grid[0])
    cols = int(grid[1])
    print('Running {} with board ({} x {}):'
          .format(SCRIPTNAME, rows, cols))
    #
    # Calculate solution maximum number of holes.
    #
    (max_holes, max_dominoes) = cmn.calc_solution_holes(rows, cols)
    print("Maximum number of holes for {} x {} board is {}."
          .format(rows, cols, max_holes))
    print("Maximum number of dominoes for {} x {} board is {}."
          .format(rows, cols, max_dominoes))
    # Determine the chromosome size 'csize' from the maximum number of rows and
    # columns that are possible; plus add some padding.
    csize = int(max_holes + max_dominoes + 4)
    print("Chromosome size = {}".format(csize))
    print("Generations={}, Population={}, Crossover Probability={}, Mutation Rate={}"
          .format(GENERATIONS_MAX, POPULATION_SIZE, CX_PROBABILITY, MUTATION_RATE))

    # Initialise the DEAP structure and evaluation object
    toolbox, pop, fits = init_DEAP_structures(csize, cols, rows, max_holes, max_dominoes)

    #
    # Run the evolution:
    # Stop if we find a solution or we reach the maximum number of generations.
    #
    # Variable keeping track of the number of generations
    generation = 0
    start = time.time() # Used to time script execution.
    while max(fits) < 100 and generation < GENERATIONS_MAX:
        # Display the header at the regular intervals.
        if not generation % 50:
            print("\n {:<10}{:<10}{:<8}{:<8}{:<8}{:<8}"
                  .format("Gen", "Eval", "Min", "Max", "Avg", "Std"))
            print("-" * 60)
        # A new generation
        random.seed(time.time())  # MUST ENSURE RANDOM SEED EACH RUN
        generation = generation + 1

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        for mutant in offspring:
            # mutate an individual with probability MUTATION_RATE
            # range [0.0, 1.0)
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        # Calculate and print out the statistics
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print(" {:<10}{:<10}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}"
              .format(generation, len(invalid_ind), min(fits), max(fits), mean, std))

    # Had to add time.sleep() to get around a weird bug, where times less than
    # 1 second are not recorded/recognised. Simple 1 second delay and then we
    # have to subtract the 1 second to get the actual time.
    time.sleep(1)
    execution_time = time.time() - start - 1
    total_solns = int(0)
    best = tools.selBest(pop, 1)[0]
#    print("Best individual is %s, %s" % (best, best.fitness.values))   # debug

    # Let the user know the result.
    if best.fitness.values[0] >= 100:
        print("\nSolution(s) Found.")
        print("Saving resultant board plots.")
        dplobj = dpl.CPlot(rows, cols, "GA", max_holes, max_dominoes)
        solns = []
        # Find all the solutions in the population as there may be more than 1
        for ind in pop:
            if ind.fitness.values[0] == 100:
                result = cmn.convert2board(ind, rows, cols)
                # Only add if the solution is unique!
                unique = True
                for soln in solns:
                    if np.array_equal(soln, result):
                        unique = False
                        break
                if unique:
#                    print("ind={}".format(ind)) # debug
                    solns.append(result)
                    # Plot the domino board solutions; saving to a each to a file.
                    # Note we need to pass in a numpy 2D array!
                    dplobj.write_soln_to_file(result)
        dplobj.classify_solns()
        dplobj.plot_all()
        #
        # Write the version information to the results folder.
        # - Python and script version information is recorded.
        # - this includes the execution time in seconds.
        #
        total_solns = len(solns)
        verstr = cmn.get_version_str(SCRIPTINFO)
        time1st = execution_time / total_solns
        dplobj.write_test_info(verstr, execution_time, time1st)
    else:
        print("\nFailed to find a solution.")

    # Display the time taken if requested.
    if timed_execution:
        print("Script execution time:", execution_time, "seconds")
    # Return the execution time.
    return execution_time, total_solns
    # Done.


if __name__ == '__main__':
    START = time.time()        # Used to time script execution.
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--version', action='version', version=SCRIPTINFO)
    PARSER.add_argument('--timer', '-t',
                        help='Script execution time.',
                        action='store_true')
    PARSER.add_argument('board', nargs=1,
                        help='Gunport problem board size, eg 10x8.')
    # Get the arguments dictionary, where arguments are the keys.
    ARGS = vars(PARSER.parse_args())
    # Extract the grid dimensions as a list [x, y]
    GRID = ARGS['board']
    GRID = GRID[0].lower().split("x")
    # Set the timer boolean value
    TIMER = ARGS['timer']

    # Start up the application
    main(GRID, TIMER)

# EOF

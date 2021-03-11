#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright(c) 2020 De Montfort University. All rights reserved.
#
#

"""
Memetic Algorithm to solve The Gun Port Problem as first described by
Sands, B. (1971), The Gunport Problem, Mathematics Magazine, Vol.44, pp.193-196
Based on the ga.py script.

This is an optimisation problem to maximise the holes and minimise the number
number of dominoes in a board of variable size. Holes are not allowed to touch
each other side on.

Command line usage example:
    py -3 ma.py 6x6
    py -3 ma.py 6x6 -s
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
import findallsolns as fas      # Find all solns from fundamental soln
import common as cmn            # Common defines and functions

__author__ = 'David Kind'
__date__ = '06-02-2020'
__version__ = '1.5'
__copyright__ = 'Copyright(c) 2020 De Montfort University. All rights reserved.'

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)

#
# MA Parameters that can be modified to alter performance
# Note: mutation rate affects the convergence speed.
#
GENERATIONS_MAX = int(2000)     # The total number of generations
POPULATION_SIZE = int(1000)     # The total number of individuals; must be even.
CX_PROBABILITY = float(0.0)     # (%) probability with which two individuals
                                # are crossed, range [0.0, 1.0)
MUTATION_RATE = float(0.25)     # (%) probability of mutating an individual,
                                # range [0.0, 1.0)

class CEvaluation():
    """
    MA Evaluation Object.
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
            # Favour a space over a domino - bias.
            result.append(cmn.CELL_SPACE)
        else:
            # Are we ok to have a horizontal domino?
            # Make sure we don't place the horizontal domino over previously placed
            # vertical domino on the board.
            if col < self.xsize - 1 and \
               self.board[row, col + 1] == cmn.CELL_SPACE:
                result.append(cmn.CELL_HDOMINO)
            # Are we ok to have a vertical domino?
            if row < self.ysize - 1:
                result.append(cmn.CELL_VDOMINO)
#        print("check_board: ({}x{}): xsize={}, ysize={}"
#              .format(row, col, self.xsize, self.ysize))   # debug
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


class CSeed():
    """
    MA Seed Object.
    Seeds a DEAP population object of n individuals with new random chromosome
    values. Seeding means injecting previous solutions' into the individual's
    chromosome in order to give it an advantage in the solution space.
    All individuals will be seeded with solutions from the (3 x 3) board and
    these will be selected randomly along with holes and dominoes.
    """
    # Define the composition of shapes to be placed on the board.
    # Defining the dimensions of the shape will make the subsequent code more
    # more flexible and adaptable if new shapes are added.
    #
    # Note: arrange shapes from smallest to largest.
    # <shape> : [m_size, n_size, [<list of holes and dominoes>]
    shapes = {
        "hole":[1, 1, [cmn.CELL_SPACE]],
        "hdom":[1, 2, [cmn.CELL_HDOMINO]],
        "vdom":[2, 1, [cmn.CELL_VDOMINO]],
        "[0]-[0]-3x3":[3, 3, [cmn.CELL_SPACE, cmn.CELL_HDOMINO, cmn.CELL_HDOMINO,
                              cmn.CELL_UNASSIGNED, cmn.CELL_UNASSIGNED, cmn.CELL_HDOMINO]],
        "[0]-[1]-3x3":[3, 3, [cmn.CELL_SPACE, cmn.CELL_VDOMINO, cmn.CELL_UNASSIGNED,
                              cmn.CELL_VDOMINO, cmn.CELL_VDOMINO, cmn.CELL_UNASSIGNED]],
        "[0]-[2]-3x3":[3, 3, [cmn.CELL_HDOMINO, cmn.CELL_UNASSIGNED, cmn.CELL_UNASSIGNED,
                              cmn.CELL_HDOMINO, cmn.CELL_HDOMINO, cmn.CELL_UNASSIGNED]],
        "[0]-[3]-3x3":[3, 3, [cmn.CELL_VDOMINO, cmn.CELL_UNASSIGNED, cmn.CELL_VDOMINO,
                              cmn.CELL_VDOMINO, cmn.CELL_UNASSIGNED, cmn.CELL_UNASSIGNED]]
    }

    def __init__(self, deap_pop, cols, rows):
        """
        :param deap_pop: DEAP population object to be seeded: list of lists.
        :param cols: number of columns for the specified board size.
        :param rows: number of rows for the specified board size.
        """
        # Initialise the object members
        self.deap_pop = deap_pop
        self.cols = cols
        self.rows = rows
        self.board = None
        self.chromosome = []
        # The deap_pop object is a list of individuals, also a list.
        # Record the size of a chromosome.
        self.ind_size = len(deap_pop[0])

    def process(self):
        """
        Process the population by seeding each individual.
        :return: n/a
        """
        # Loop through the entire population, generating each individual.
        for ind in self.deap_pop:
            # Seed an individual, don't need the old one so can ignore.
            chromosome = self.seed_individual()
            # Add the new individual chromosome to the population
            for idx in range(self.ind_size):
                ind[idx] = chromosome[idx]
#### TODO: DEBUG ONLY.................................................................................................
#            print("\n" + "*" * 80)
#            print("chromosome = {}".format(chromosome))
#            print("*" * 80)
#            break  ### TODO: remove me: debug only......................................................................
#### TODO: DEBUG ONLY.................................................................................................

    def seed_individual(self):
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
        # Initialise a blank chromosome
        chromosome = [cmn.CELL_UNASSIGNED] * self.ind_size
        # Initialise an empty board
        self.board = np.full((self.rows, self.cols), cmn.CELL_UNASSIGNED, 'uint8')
        idx = 0  # Chromosome index
        for row in range(self.rows):
            for col in range(self.cols):
#### TODO: DEBUG ONLY.................................................................................................
#                print("\n" + "-" * 80)
#                print("board[{}, {}] = \n{}".format(row, col, self.board))
#### TODO: DEBUG ONLY.................................................................................................
                #
                # Is the board location already occupied
                #
                if self.board[row, col] != cmn.CELL_UNASSIGNED:
                    # It is occupied move along to the next position
                    continue
                # Create a list of shapes that can be placed at this location.
                # Ref: self.shapes
                # ok_shapes is a list of keys from self.shapes
                ok_shapes = self.check_board(row, col)
                num_ok_shapes = len(ok_shapes)
#                print("ok_shapes={}".format(ok_shapes))                 # debug
                #
                # Randomly select an OK shape and place it; if we have a list
                # of OK shapes that is...
                #
                if num_ok_shapes:
                    # Need to randomly select a shape to place in this location
                    # from the shapes that are available.
                    key = ok_shapes[random.randint(0, num_ok_shapes - 1)]
                    self.set_board(self.board, row, col, key)
#                    print("Selected shape = {}".format(key))            # debug
                # Nothing is OK, but we have a space either above or to
                # the left so can replace that with a domino.
                else:
                    if col > 0 and self.board[row, col - 1] == cmn.CELL_SPACE:
                        # Replace space with Horizontal domino
                        self.board[row, col - 1] = cmn.CELL_HDOMINO
                        self.board[row, col] = cmn.CELL_HDOMINO
                    else:
                        # Replace space with Vertical domino
                        if row > 0 and self.board[row - 1, col] == cmn.CELL_SPACE:
                            self.board[row - 1, col] = cmn.CELL_VDOMINO
                            self.board[row, col] = cmn.CELL_VDOMINO
                        else:
                            print("ERROR: UNABLE TO PLACE SHAPE!!!")
#                    print("=> nothing ok")                              # debug
                    # Don't increase the idx counter as we may to re-evaluate
                    # the shape currently being pointed at.
                    continue
                # Look at the next chromosome element
                idx += 1
        # Return the newly generated individual chromosome.
        return cmn.convert2chrom(self.board, self.ind_size)

    def check_board(self, row, col):
        """
        Returns a list of valid shape keys, after verifying whether the shape
        can be fitted to the specified location in the board. Only checks the
        current location against proceeding locations and not the next
        location.
        Shapes are: ref: self.shapes
        :param row: board row position.
        :param col: board column position.
        """
#### TODO: DEBUG ONLY.................................................................................................
#        print("+" * 80)
#### TODO: DEBUG ONLY.................................................................................................
        # Calculate the dimensions of the square space remaining from this
        # location. This information can be used to decide which shapes can be
        # placed here.
        (rows_max, cols_max) = np.shape(self.board)
        rows_size = rows_max - row # number of rows available
        cols_size = cols_max - col # number of cols available
        valid_shapes = []
#### TODO: DEBUG ONLY.................................................................................................
#        print("1) rows_size={}, cols_size={}".format(rows_size, cols_size))
#### TODO: DEBUG ONLY.................................................................................................
        # Check to see if the square space is empty and if not adjust it.
        for srow in range(row, rows_max):
            for scol in range(col, cols_max):
#### TODO: DEBUG ONLY.................................................................................................
#                print("board[{}][{}] = {}".format(srow, scol, self.board[srow][scol]))
#### TODO: DEBUG ONLY.................................................................................................
                if self.board[srow][scol] != cmn.CELL_UNASSIGNED:
                    # Found an occupied square.
                    if scol != col and (scol - col) < cols_size:
                        # Adjust the column size accordingly
                        cols_size = scol - col
                    if srow != row and (srow - row) < rows_size:
                        # Adjust the row size accordingly
                        rows_size = srow - row
#### TODO: DEBUG ONLY.................................................................................................
#        print("2) rows_size={}, cols_size={}".format(rows_size, cols_size))
#### TODO: DEBUG ONLY.................................................................................................
        # Loop through all shapes adding those that will fit
        for skey in self.shapes.keys():
            if self.shapes[skey][0] <= rows_size and self.shapes[skey][1] <= cols_size:
                valid_shapes.append(skey)
#### TODO: DEBUG ONLY.................................................................................................
#        print("3) valid_shapes={}".format(valid_shapes))
#### TODO: DEBUG ONLY.................................................................................................
        # We have a list of shapes that will fit, but do they fit without
        # causing an error. For example, hole next to hole.
        # Loop through the list and remove those that are not valid.
        # Basically making sure that holes are not touching side on.
        vshapes = valid_shapes.copy()
        valid_shapes = []
        for skey in vshapes:
            board = self.board.copy()
            self.set_board(board, row, col, skey)
            shape_ok = True
            if row > 0:
                for ccol in range(col, col +  self.shapes[skey][1]):
                    if board[row, ccol] == cmn.CELL_SPACE and \
                       board[row - 1, ccol] == cmn.CELL_SPACE:
                        shape_ok = False
            if col > 0:
                for crow in range(row, row +  self.shapes[skey][0]):
                    if board[crow, col] == cmn.CELL_SPACE and \
                       board[crow, col - 1] == cmn.CELL_SPACE:
                        shape_ok = False
            if shape_ok:
                valid_shapes.append(skey)
#### TODO: DEBUG ONLY.................................................................................................
#        print("4) valid_shapes={}".format(valid_shapes))
#        print("+" * 80)
#### TODO: DEBUG ONLY.................................................................................................
        #
        # Add bias in order: shapes, holes, dominoes
        #
        if len(valid_shapes) > 1:
            vshapes = valid_shapes.copy()
            valid_shapes = []
            for vshape in vshapes:
                if vshape != "hole" and vshape != "hdom" and vshape != "vdom":
                    # Bias towards the larger shapes
                    valid_shapes.append(vshape)
                    valid_shapes.append(vshape)
                    valid_shapes.append(vshape)
                    valid_shapes.append(vshape)
                elif vshape == "hole":
                    # Add bias towards holes
                    valid_shapes.append(vshape)
                # Make sure the dominoes get added!
                valid_shapes.append(vshape)
#### TODO: DEBUG ONLY.................................................................................................
#        print("5) valid_shapes={}".format(valid_shapes))
#        print("+" * 80)
#### TODO: DEBUG ONLY.................................................................................................
        return valid_shapes

    def set_board(self, brd, row, col, key):
        """
        Set the contents of the current board using the shape key.
        Assumes the board has been zero'd before use.
        :param brd: board to set.
        :param row: board row position.
        :param col: board column position.
        :param key: key to self.shapes dictionary.
        """
        # Note: arrange shapes from smallest to largest.
        # <shape> : [m_size, n_size, [<list of holes and dominoes>]
        row_size = self.shapes[key][0]
        col_size = self.shapes[key][1]
        chrom = self.shapes[key][2]
        sboard = cmn.convert2board(chrom, row_size, col_size)
        for srow in range(row_size):
            for scol in range(col_size):
                brd[row + srow][col + scol] = sboard[srow][scol]

    def get_population(self):
        """
        Return a reference to the seeded population.
        :return: Population, list of lists.
        """
        return self.deap_pop


def init_DEAP_structures(chromosome_size, cols, rows, max_holes, max_dominoes, popSeed=False):
    """
    Initialises the DEAP framework for GA and MA use.
    :param chromosome_size: the maximum size of the chromosome.
    :param cols: number of columns for the specified board size.
    :param rows: number of rows for the specified board size.
    :param max_holes: solution maximum number of holes.
    :param max_dominoes: solution minimum number of dominoes.
    :param popSeed: boolean flag used to seed the population with shapes.
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
    # Create an initial population of n individuals
    # (where each individual is a list of integers).
    pop = toolbox.population(n=POPULATION_SIZE)
    if popSeed:
        # Need to override the initial DEAP population and seed it manually.
        seed = CSeed(pop, cols, rows)
        print("Seeding the population...")
        seed.process()
        pop = seed.get_population()
#    print("Start of evolution")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("\nEvaluating {} individuals".format(len(pop)))
    # Extracting all the fitness values into a list
    fits = [ind.fitness.values[0] for ind in pop]
    return toolbox, pop, fits


def main(grid, timed_execution, seed_population):
    """
    Main Memetic Algorithm implementation.
    :param timed_execution: flag to time algorithm execution.
    :param seed_population: flag to seed population with shapes.
    :return: execution time and total number of solutions found.
    """
    # Setup the MA configuration and show the user
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
    toolbox, pop, fits = init_DEAP_structures(csize, cols, rows, max_holes, max_dominoes, seed_population)
#### TODO: remove me: stop the code running further while we are developing.
#    sys.exit(0)
#### TODO: remove me: stop the code running further while we are developing.
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
        dplobj = dpl.CPlot(rows, cols, "MA", max_holes, max_dominoes)
        solns = []  # All the solutions found
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
                    #
                    # Identify all the solutions from the fundamental solution.
                    # Takes the solution board as an input, this is a numpy ndarray and then
                    # performs rotations and flips to extract all the possible solutions.
                    #
                    all_solns = fas.findall(result)
                    # Now append all the solutions found to the end of the total solutions list.
                    solns.extend(all_solns)
                    # Plot the domino board solutions; saving to a each to a file.
                    # Note we need to pass in a numpy 2D array!
                    for result in all_solns:
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
        print("-" * 80)
        print("\nFailed to find a solution.")
        print("Looking for holes={}, dominoes={}".format(max_holes, max_dominoes))
        print("Best individual found is:")
        evalobj = CEvaluation(cols, rows, max_holes, max_dominoes)
        evalobj.eval_fitness(best)
        print(evalobj)
        # Create a plot and write the best individual to file.
        dplobj = dpl.CPlot(rows, cols, "MA", evalobj.spaces, evalobj.dominoes)
        result = cmn.convert2board(best, rows, cols)
        #
        # Identify all the solutions from the fundamental solution.
        # Takes the solution board as an input, this is a numpy ndarray and then
        # performs rotations and flips to extract all the possible solutions.
        #
        all_solns = fas.findall(result)
        # Plot the domino board solutions; saving to a each to a file.
        # Note we need to pass in a numpy 2D array!
        for result in all_solns:
            dplobj.write_soln_to_file(result)
        dplobj.classify_solns()
        dplobj.plot_all()

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
    PARSER.add_argument('--seed', '-s',
                        help='Seed the population.',
                        action='store_true')
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
    # Set the timer boolean flag
    TIMER = ARGS['timer']
    # Set the seed boolean flag
    SEED = ARGS['seed']

    # Start up the application
    main(GRID, TIMER, SEED)

# EOF

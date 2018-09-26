#!/usr/bin python
# -*- coding: utf-8 -*-
#
# Copyright 2018 David Kind
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# This template script comes with command line interface and logging.
# Schlumberger Private

'''
Embedded Computing Challenge 27.
What is the maximum number of 1 x 1 holes that can be obtained by arranging
dominoes on an m x n field ?
Ref: Challenge_Problem-27.docx
     https://www.yammer.com/slb.com/threads/1124769403

Solution uses a Genetic Algorithm to optimise the problem.
'''

# TODO:
# 1) Fix the domino colours, it look wrong.
# 2) Add a graph to show the progress.
# 3) Fix the cross-over so that it is alternating; does this make a difference.

import os
import sys
import platform
import argparse
import time                     # Used to time script execution.
import math
from random import randint      # Used to generate random integer values.
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from filterpy.monte_carlo import stratified_resample
import numpy as np

__author__ = 'David Kind'
__date__ = '26-09-2018'
__version__ = '1.4'
__copyright__ = 'http://www.apache.org/licenses/LICENSE-2.0'


if not hasattr(sys, "hexversion") or sys.hexversion < 0x02070600:
    print 'Python version:', platform.python_version(), \
        '- I need python 2.7.6 or greater'
    print 'http://www.python.org/download/releases/2.7.6/'
    exit(1)

#
# Main script defines
#
SCRIPTNAME = os.path.basename(sys.argv[0])
SCRIPTINFO = "{} version: {}, {}".format(SCRIPTNAME, __version__, __date__)

#
# Grid defines; modify these values to change the grid size to optimise.
#
X_WIDTH = int(10)
Y_HEIGHT = int(8)
GRID_SIZE = X_WIDTH * Y_HEIGHT

#
# GA Parameters that can be modified to alter performance
#
INDIVIDUAL_MUTATION_RATE = float(0.10)  # (%) of individual to mutate
POPULATION_MUTATION_RATE = float(0.5)   # (%) of population to mutate
POPULATION_SIZE = int(1000)             # The total number of individuals
                                        # Must be an even number.

GENERATIONS_MAX = int(1000)             # The total number of generations
NUM_SHAPES = 3              # Defined shapes
GRID_SPACE = 0              # Empty square
GRID_HDOMINO = 128          # Horizontal domino
GRID_VDOMINO = 255          # Vertical domino


class CIndividual(object):
    '''
    GA Individual Object.
    '''
    gene = []           # list of shapes; these are fitted to the grid
    weight = float()    # (%) individual weight/fitness rating
    grid = None         # Resultant grid with domnoes placed
    spaces = int(0)     # Number of empty spaces in the fitted grid
    dominoes = int(0)   # Number of domonies placed in the fitted grid

    def __init__(self):
        # Initialise the object members
        self.gene = [randint(0, NUM_SHAPES - 1) for _ in range(GRID_SIZE)]
        self.fit_to_grid()
        # Set an initial weighting; we'll be using the number of spaces in
        # each resultant grid for the weighting calculation. This means grids
        # with a greater number of spaces will have a greater weighting. The
        # weighting will have to be normalised after each run.
        self.weight = float(1) / float(POPULATION_SIZE)

    def mutate(self):
        '''
        Mutate the gene by the mutation rate.
        Calculates the individual mutation as a percentage of the overall size
        of the gene and rounds the result up to ensure at least one entry is
        always modified.
        '''
        # Determine the number of gene entries need to be mutated
        gene_len = len(self.gene)
        num_entries = int(math.ceil(gene_len * INDIVIDUAL_MUTATION_RATE))
        # Mutate the random locations
        gene_len -= 1
        for _ in range(num_entries):
            index = randint(0, gene_len)
            # Toggle the gene entry for our mutation
            new_shape = randint(0, NUM_SHAPES - 1)
            self.gene[index] = new_shape
        self.fit_to_grid()

    def fit_to_grid(self):
        '''
        Fits the gene to the grid correcting any issues so that the resulting
        pattern is acceptable; this could mean replacing/removing/adding some
        of the shapes. The resultant grid contains only 1s and 0s, where 1
        represents a dominoe and 0 represents a hole.
        Check for:
          (1) empty spaces next to each other, testing above and left only.
          (2) replacing overlapping dominoes.
          (3) removing unused genes from the end of the list.
        '''
        self.grid = np.zeros((Y_HEIGHT, X_WIDTH))
        self.spaces = int(0)
        self.dominoes = int(0)
        idx = 0
        for row in range(Y_HEIGHT):
            for col in range(X_WIDTH):
                value = self.gene[idx]
                #
                # Is the grid location already occupied
                #
                if self.grid[row, col] != GRID_SPACE:
                    # It is occupied move along to the next position
                    continue
                # Set up some booleans to help us make decisions on the placement
                # Are we ok to place have an empty space?
                if (col > 0 and self.grid[row, col - 1] == GRID_SPACE) or \
                    (row > 0 and self.grid[row - 1, col] == GRID_SPACE):
                    space_ok = False
                else:
                    space_ok = True
                # Are we ok to have a horizontal domino?
                if col < X_WIDTH - 1 and self.grid[row, col + 1] == GRID_SPACE:
                    hdomino_ok = True
                else:
                    hdomino_ok = False
                # Are we ok to have a vertical domino?
                if row < Y_HEIGHT - 1:
                    vdomino_ok = True
                else:
                    vdomino_ok = False
                #
                # Set the contents of the current grid location using the
                # stored value if possible.
                #
                if value == GRID_SPACE and space_ok:
                    # Empty space is OK
                    self.spaces += 1
                elif value == GRID_HDOMINO and hdomino_ok:
                    # Horizontal domino OK
                    self.grid[row, col] = GRID_HDOMINO
                    self.grid[row, col + 1] = GRID_HDOMINO
                    self.dominoes += 1
                elif value == GRID_VDOMINO and vdomino_ok:
                    # Vertical domino OK
                    self.grid[row, col] = GRID_VDOMINO
                    self.grid[row + 1, col] = GRID_VDOMINO
                    self.dominoes += 1
                else:
                    # Need to try and fit a shape to this location.
                    # The current value (shape) does not fit.
                    # The preference is to select a space if ok to do so.
                    # If a space is not OK then we need to randomly select a
                    # domino if possible.
                    if space_ok:
                        self.spaces += 1
                        self.gene[idx] = GRID_SPACE
                    elif hdomino_ok or vdomino_ok:
                        # If both domino types are ok then we need to randomly
                        # select one to avoid any kind of resultant bias.
                        # This is done by forcing only one of the dominoes to
                        # to be OK.
                        if hdomino_ok and vdomino_ok:
                            if randint(0,1):
                                hdomino_ok = False
                            else:
                                vdomino_ok = False
                        if hdomino_ok:
                            # Horizontal domino OK
                            self.gene[idx] = GRID_HDOMINO
                            self.grid[row, col] = GRID_HDOMINO
                            self.grid[row, col + 1] = GRID_HDOMINO
                            self.dominoes += 1
                        else: # vdomino_ok:
                            # Vertical domino OK
                            self.gene[idx] = GRID_VDOMINO
                            self.grid[row, col] = GRID_VDOMINO
                            self.grid[row + 1, col] = GRID_VDOMINO
                            self.dominoes += 1
                    else:
                        # Nothing is OK, but we have a space either above or to
                        # the left so can replace that with a domino. This is
                        # in all likelyhood the last grid square.
                        self.spaces -= 1
                        self.dominoes += 1
                        if col > 0 and self.grid[row, col - 1] == GRID_SPACE:
                            # Horizontal domino
                            self.gene[idx - 1] = GRID_HDOMINO
                            self.grid[row, col - 1] = GRID_HDOMINO
                            self.grid[row, col] = GRID_HDOMINO
                        else:
                            # Vertical domino
                            # Update the gene with a vertical domino, this is
                            # a little more tricky than simply adding a
                            # horizontal domino, we have to scan back to the
                            # last vertical domino and replace the space with a
                            # vertical domino.
                            for idxnew in range(idx, -1, -1):
                                if self.gene[idxnew] == GRID_VDOMINO:
                                    self.gene[idxnew + 1] = GRID_VDOMINO
                                    break
                            self.grid[row - 1, col] = GRID_VDOMINO
                            self.grid[row, col] = GRID_VDOMINO
                idx += 1 # Look at the next gene element

    def __str__(self):
        # Return a the resultant order n checker-board
        outstr = "\n{}\nfitness={}%".format(self.gene, self.weight)
        outstr += ", spaces={}".format(self.spaces)
        outstr += "\n{}".format(self.grid)
        return outstr

def calcSolutionHoles(x, y):
    '''
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
    '''
    holes = int(0)
    # 1) Are x or y divisible by 3
    if not (x % 3) or not (y % 3):
        holes = (x * y) / 3
    else:
        # Note: to get here we know that neither x or y are divisible by 3.
        x_const = x - (int(x / 3) * 3)
        y_const = y - (int(y / 3) * 3)
        # 2) If x or y both have equal constants
        if x_const == y_const:
            holes = (x * y - 4) / 3
        # 3) If x or y both have unequal constants
        else:
            holes = (x * y - 2) / 3
    return int(holes)


def main(time_execution):
    '''Main function'''
    START = time.time()        # Used to time script execution.
    print 'Running {} with grid ({} x {}):'.format(SCRIPTNAME, X_WIDTH, Y_HEIGHT)

    #
    # Calculate solution maximum number of holes.
    #
    maxHoles = calcSolutionHoles(X_WIDTH, Y_HEIGHT)
    print "Maximum number of holes for {} x {} grid is {}." \
            .format(X_WIDTH, Y_HEIGHT, maxHoles)

    #
    # Create our initial random population and fit to grid
    #
    print "Creating the initial population."
    population = [CIndividual() for _ in range(POPULATION_SIZE)]

    #
    # Run GA over GENERATIONS_MAX for each individual
    #
    print "Executing the GA over {} generations.".format(GENERATIONS_MAX)
    most_spaces = 0
    firsthalf = GRID_SIZE / 2
    for x in range(GENERATIONS_MAX):
        total_weighting = 0
        for individual in range(POPULATION_SIZE):
            # Mutate the individual
            population[individual].mutate()
            total_weighting += population[individual].spaces
            # Set the best individual
            if population[individual].spaces > most_spaces:
                best = deepcopy(population[individual])
                most_spaces = population[individual].spaces
#                print best

        # Quit if the problem has been solved.
        if most_spaces == maxHoles:
            break

        # We need to normalise all the population weights.
        weights = []
        for individual in range(POPULATION_SIZE):
            population[individual].weight = float(population[individual].spaces) / total_weighting
            weights.append(population[individual].weight)
        # Now resample the best performing individuals using Monte Carlo
        # stratified resampling.
        # Ref: https://filterpy.readthedocs.io/en/latest/index.html
        indexes = stratified_resample(weights)
        # Create our new population
        nwpop = []  # nwpop (new population)
        for individual in range(0, POPULATION_SIZE, 2):
            # Copy across 2 at a time; we're going to apply cross-over.
            nwpop.append(deepcopy(population[indexes[individual]]))
            nwpop.append(deepcopy(population[indexes[individual + 1]]))
            # Gene cross-over with resampled individuals
            g1 = nwpop[individual].gene[:firsthalf] + nwpop[individual + 1].gene[firsthalf:]
            g2 = nwpop[individual + 1].gene[:firsthalf] + nwpop[individual].gene[firsthalf:]
            nwpop[individual].gene = g1
            nwpop[individual + 1].gene = g2

        # Create a new copy of the next generation
        population = deepcopy(nwpop)
        print ".",  # Show progress; that the script is still running.

    # Let the user know the result.
    if most_spaces == maxHoles:
        print "\nSolution Found."
    else:
        print "\nFailed to find a solution."

    # Are we on the timer?
    if time_execution:
        print "Script execution time:", time.time() - START, "seconds"
    #
    # Display the domino grid; it's a bit rough and ready.
    #
    fig, ax = plt.subplots()
    ax.imshow(best.grid, cmap=cm.jet, interpolation='nearest')
    ax.set_xticks([x + 0.5 for x in range(X_WIDTH)])
    ax.set_yticks([y + 0.5 for y in range(Y_HEIGHT)])
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    msg = "({} x {}) {} spaces, {} dominoes" \
        .format(X_WIDTH, Y_HEIGHT, best.spaces, best.dominoes)
    plt.title(msg)
    plt.show()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     version=SCRIPTINFO,
                                     formatter_class=argparse.RawTextHelpFormatter)
    PARSER.add_argument('--timer', '-t',
                        help='Script execution time.',
                        action='store_true')
    # Get the arguments dictionary, where arguments are the keys.
    ARGS = vars(PARSER.parse_args())

    # Start up the application
    main(ARGS['timer'])

# EOF


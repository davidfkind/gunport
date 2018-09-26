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
Genetic Algorithm to solve The Gun Port Problem as first described by
Sands, B. (1971), The Gunport Problem, Mathematics Magazine, Vol.44, pp.193-196

This is an optimisation problem to maximise the holes and minimise the number
number of dominoes in a grid of variable size. Holes are not allowed to touch
each other side on.
'''

# TODO:
# 1) Fix the domino colours, they look wrong.                                   (only if I have time)
# 3) Fix the cross-over so that it is alternating; does this make a difference.
#    Could add different methods and compare them.
# 4) Modify the script to use a preference for the domino/hole placement.
#    Use this code to remove the population creation biases.
# 5) Check the mutation code.
#


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
__version__ = '1.5'
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
Y_HEIGHT = int(6)
GRID_SIZE = X_WIDTH * Y_HEIGHT

#
# GA Parameters that can be modified to alter performance
#
INDIVIDUAL_MUTATION_RATE = float(0.10)  # (%) of individual to mutate
POPULATION_MUTATION_RATE = float(0.5)   # (%) of population to mutate
POPULATION_SIZE = int(100)              # The total number of individuals
                                        # Must be an even number.

GENERATIONS_MAX = int(1000)             # The total number of generations
NUM_SHAPES = 3                          # Defined shapes
GRID_SPACE = 0                          # Empty square
GRID_HDOMINO = 128                      # Horizontal domino
GRID_VDOMINO = 255                      # Vertical domino


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
        Fits the gene to a blank grid correcting any issues so that the
        resulting pattern is acceptable; this could mean replacing/removing or
        adding some of the shapes. The resultant grid contains only 0s and
        values > 1, where values > 1 represent a domino and 0 represents a hole.
        Check for:
          (1) empty spaces next to each other, testing above and left only.
          (2) replacing overlapping dominoes.
          (3) removing unused genes from the end of the list.
        '''
        self.grid = np.zeros((Y_HEIGHT, X_WIDTH))
        self.spaces = int(0)    # Running total of spaces in grid
        self.dominoes = int(0)  # Running total of dominoes in grid
        idx = 0                 # Gene index
        for row in range(Y_HEIGHT):
            for col in range(X_WIDTH):
                #
                # Is the grid location already occupied
                #
                if self.grid[row, col] != GRID_SPACE:
                    # It is occupied move along to the next position
                    continue
                # Set up some booleans to help us make decisions on the
                # placement. The booleans represent the 3x types:
                # hole, vertical domino and horizontal domino. If True then
                # this shape can be fitted to the specified location in the
                # grid.
                sp_ok, hd_ok, vd_ok = self.check_grid(row, col)
                # Create a list of shapes that can be placed at this location.
                ok_shapes = []
                if sp_ok:
                    ok_shapes.append(GRID_SPACE)
                if hd_ok:
                    ok_shapes.append(GRID_HDOMINO)
                if vd_ok:
                    ok_shapes.append(GRID_VDOMINO)
                #
                # Set the contents of the current grid location using the
                # stored value if possible.
                #
                num_ok_shapes = len(ok_shapes)
                if self.gene[idx] in ok_shapes:
                    self.set_grid(row, col, self.gene[idx])
                # The current value (shape) does not fit.
                # Need to try and fit a shape to this location and other shapes
                # are able to fit in this grid location.
                elif num_ok_shapes:
                    # Need to randomly select a shape to place in this location
                    # from the shapes that are available.
                    val = ok_shapes[randint(0, num_ok_shapes - 1)]
                    self.set_grid(row, col, val)
                # Nothing is OK, but we have a space either above or to
                # the left so can replace that with a domino. This is
                # in all likelyhood the last grid square.
                else:
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

    def check_grid(self, row, col):
        '''
        Returns 3x booleans which represent the 3x types: hole, vertical domino
        and horizontal domino. If True then this shape can be fitted to the
        specified location in the grid.
        sp_ok (space/hole):         True=OK to fit to grid; False=Not OK to fit.
        hd_ok (horizontal domino):  True=OK to fit to grid; False=Not OK to fit.
        vd_ok (vertical domino):    True=OK to fit to grid; False=Not OK to fit.
        '''
        # Are we ok to place have an empty space?
        if (col > 0 and self.grid[row, col - 1] == GRID_SPACE) or \
            (row > 0 and self.grid[row - 1, col] == GRID_SPACE):
            sp_ok = False
        else:
            sp_ok = True
        # Are we ok to have a horizontal domino?
        if col < X_WIDTH - 1 and self.grid[row, col + 1] == GRID_SPACE:
            hd_ok = True
        else:
            hd_ok = False
        # Are we ok to have a vertical domino?
        if row < Y_HEIGHT - 1:
            vd_ok = True
        else:
            vd_ok = False
        return sp_ok, hd_ok, vd_ok

    def set_grid(self, row, col, value):
        '''
        Set the contents of the current grid location using the value.
        '''
        if value == GRID_SPACE:
            # Empty space is OK
            self.spaces += 1
        elif value == GRID_HDOMINO:
            # Horizontal domino OK
            self.grid[row, col] = GRID_HDOMINO
            self.grid[row, col + 1] = GRID_HDOMINO
            self.dominoes += 1
        else: # value == GRID_VDOMINO:
            # Vertical domino OK
            self.grid[row, col] = GRID_VDOMINO
            self.grid[row + 1, col] = GRID_VDOMINO
            self.dominoes += 1

    def display_grid(self):
        '''
        Displays the resultant grid showing the holes and dominoes of the
        individual.
        '''
        _, ax = plt.subplots()
        ax.imshow(self.grid, cmap=cm.jet, interpolation='nearest')
        ax.set_xticks([x + 0.5 for x in range(X_WIDTH)])
        ax.set_yticks([y + 0.5 for y in range(Y_HEIGHT)])
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        msg = "({} x {}) {} spaces, {} dominoes" \
            .format(X_WIDTH, Y_HEIGHT, self.spaces, self.dominoes)
        plt.title(msg)
        plt.show()

    def __str__(self):
        # Return a the resultant order n checker-board
        outstr = "\n{}\nfitness={}%".format(self.gene, self.weight)
        outstr += ", spaces={}".format(self.spaces)
        outstr += "\n{}".format(self.grid)
        return outstr

def calc_solution_holes(x_coord, y_coord):
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
    # 1) Are x_coord or y_coord divisible by 3
    if not x_coord % 3 or not y_coord % 3:
        holes = (x_coord * y_coord) / 3
    else:
        # Note: to get here we know that neither x_coord or y_coord are divisible by 3.
        x_const = x_coord - (int(x_coord / 3) * 3)
        y_const = y_coord - (int(y_coord / 3) * 3)
        # 2) If x_coord or y_coord both have equal constants
        if x_const == y_const:
            holes = (x_coord * y_coord - 4) / 3
        # 3) If x_coord or y_coord both have unequal constants
        else:
            holes = (x_coord * y_coord - 2) / 3
    return int(holes)

def plot_performance(mse_values):
    '''
    Plot of the performance of the algorithm.
    mse_values is a list of calculated MSE values for each iterated generation.
    '''
    plt.plot(mse_values)
    plt.ylabel('MSE')
    plt.xlabel('Generations')
    plt.show()


def main(time_execution):
    '''Main function'''
    start = time.time()        # Used to time script execution.
    print 'Running {} with grid ({} x {}):'.format(SCRIPTNAME, X_WIDTH, Y_HEIGHT)

    #
    # Calculate solution maximum number of holes.
    #
    max_holes = calc_solution_holes(X_WIDTH, Y_HEIGHT)
    print "Maximum number of holes for {} x {} grid is {}." \
            .format(X_WIDTH, Y_HEIGHT, max_holes)

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
    mse_values = []
    mse_values.append(max_holes ** 2)
    firsthalf = GRID_SIZE / 2
    for _ in range(GENERATIONS_MAX):
        mse = 0.0
        total_weighting = 0
        for individual in range(POPULATION_SIZE):
            # Mutate the individual
            population[individual].mutate()
            total_weighting += population[individual].spaces
            # Set the best individual
            if population[individual].spaces > most_spaces:
                best = deepcopy(population[individual])
                most_spaces = population[individual].spaces
                print best
            # Calculate the mse running total
            mse += (max_holes - population[individual].spaces) ** 2

        # Now calculate and store the Mean Squared Error
        mse = mse / POPULATION_SIZE
        mse_values.append(mse)
        # Quit if the problem has been solved.
        if most_spaces == max_holes:
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
    if most_spaces == max_holes:
        print "\nSolution Found."
    else:
        print "\nFailed to find a solution."

    # Are we on the timer?
    if time_execution:
        print "Script execution time:", time.time() - start, "seconds"
    #
    # Display the domino grid; it's a bit rough and ready.
    #
    best.display_grid()
    #
    # Display the GAs overall performance.
    #
    plot_performance(mse_values)
    # DONE.


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

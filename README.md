# gunport
Genetic Algorithm to solve the Gun Port Problem

Optimization problem to determine the maximum number of holes in a grid (m x n) using dominoes to separate them.
The holes may only touch on the corners and must not touch on the sides.

The script is written in Python 2.7 and implements a genetic algorithm to solve this optimization problem. A gene represents how a grid is populated, where each element of the gene represents either: a hole, a vertical domino or a horizontal domino.
The grid is filled starting at the top left square and then each square from left to right, moving down a row when the last square on the right hand side of the row is filled.
The example.png file shows how the grid is filled and the gene is represented.

The genetic algorithm uses a population size of 100 and the gene size is made the same as the number of squares in a grid to ensure full coverage. The weighting for each individual is based on the number of spaces contained in the grid. The algorithm uses Monte Carlo stratified resampling to select the individuals to be used to create the next population, where those individuals with more holes are more likely to be selected.

The original algorithm seems to work really well, however it does have a preference for horizontal dominoes. This means all testing was carried out so that the x axis was always equal to or greater than the y axis.

Reference: Sands, B (1971), The Gunport Problem, Mathematics Magazine, Vol.44, pp. 193-196

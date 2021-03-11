py -3 .\tune_ga.py Mutation -m 0.01 -p 100 -g 1000 > tmp-mut_0.01.log
py -3 .\tune_ga.py Mutation -m 0.25 -p 100 -g 1000 > tmp-mut_0.25.log
py -3 .\tune_ga.py Mutation -m 0.50 -p 100 -g 1000 > tmp-mut_0.50.log
py -3 .\tune_ga.py Mutation -m 0.75 -p 100 -g 1000 > tmp-mut_0.75.log
py -3 .\tune_ga.py Mutation -m 0.99 -p 100 -g 1000 > tmp-mut_0.99.log

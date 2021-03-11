py -3 .\tune_ga.py Population -p 750 > tmp-750.log
py -3 .\tune_ga.py Population -p 1000 > tmp-1000.log
py -3 .\tune_ga.py Crossover -c 0.01 > tmp-cx_0.01.log
py -3 .\tune_ga.py Crossover -c 0.25 > tmp-cx_0.25.log
py -3 .\tune_ga.py Crossover -c 0.50 > tmp-cx_0.50.log
py -3 .\tune_ga.py Crossover -c 0.75 > tmp-cx_0.75.log
py -3 .\tune_ga.py Crossover -c 0.99 > tmp-cx_0.99.log

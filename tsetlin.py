import math

from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np

data = np.loadtxt("parsed_games.txt")
# Questions
X = data[:, 0:-1]
# Answers
Y = data[:, -1]

split = 0.8
z = math.floor(len(data)*0.8)

X_train = X[:z]
Y_train = Y[:z]

X_test = X[z:]
Y_test = Y[z:]


tm = MultiClassTsetlinMachine(number_of_clauses=1000, T=40, s=4)

for x in range(500):
    tm.fit(X_train, Y_train, incremental=True, epochs=1)
    print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())
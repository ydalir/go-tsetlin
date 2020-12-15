import math
import pickle
import matplotlib.pyplot as plt

from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np

data = np.loadtxt("out_all_5k.txt")
np.random.shuffle(data)
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


tm = MultiClassTsetlinMachine(number_of_clauses=100, T=20, s=7.0)

total_acc = []

print("Starting training")
for x in range(100):
    tm.fit(X_train, Y_train, incremental=True, epochs=1)
    accuracy = 100*(tm.predict(X_test) == Y_test).mean()
    print("Accuracy test:", accuracy)
    total_acc.append(accuracy)
    accuracy = 100*(tm.predict(X_train) == Y_train).mean()
    print("Accuracy train:", accuracy)

plt.plot(total_acc)
plt.show()
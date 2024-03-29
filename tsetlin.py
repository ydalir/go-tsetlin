import math
import matplotlib.pyplot as plt

from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np

data = np.loadtxt("250_10k.txt")
np.random.seed(20)
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

epochs = 50
tm = MultiClassTsetlinMachine(number_of_clauses=1000, T=25, s=6)

info = "; ".join(f"""
Clauses: {tm.number_of_clauses}
T: {tm.T}
s: {tm.s}
Epochs: {epochs}
Moves: 250
Games: {data.shape[0]}
""".strip().split("\n"))

total_acc = []

print(info)
print("Starting training")
for x in range(epochs):
    print("Epoch: ", x)
    tm.fit(X_train, Y_train, incremental=True, epochs=1)
    accuracy = 100*(tm.predict(X_test) == Y_test).mean()
    print("Accuracy test:", accuracy)
    total_acc.append(accuracy)
    accuracy = 100*(tm.predict(X_train) == Y_train).mean()
    print("Accuracy train:", accuracy)

plt.plot(total_acc)
plt.title(info)
plt.show()
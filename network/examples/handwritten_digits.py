# neural network 
from ..models import Layer, NeuralNetwork
from ..activitions import *

# for plotting
import matplotlib.pyplot as plt
import numpy as np

# datasets
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle


print('fetching...')
mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

x, t = mnist
x = x / np.max(x)

x, t = shuffle(x, t, random_state=42)

# converted it for one-hot datapoints
y = []
for n in t:
    nodes = [0] * 10
    nodes[int(n)] = 1
    y.append(nodes)
y = np.array(y, dtype=float)

print("[mnist] is fetched.")

# neural network setup
nn = NeuralNetwork(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                   Layer(784),
                   Layer(16, ReLU),
                   Layer(16, ReLU),
                   Layer(10, softmax),
                   cost= CCE,
                   optimizer= Adam(β1=0.9, β2=0.99, lr=0.05)
                   )

# data split
n = int(0.85 * len(x))
batch_size, epochs = 325, 8
xtrain, ytrain, ttrain, xtest, ytest, ttest = x[:n], y[:n], t[:n], x[n:], y[n:], t[n:]  

# train
print('training...')
loss1, acc1 = nn.learn(xtrain, ytrain, ttrain, lr=0.01, epochs=epochs, batch_size=batch_size)

# test
print("testing...")
loss2, acc2 = nn.test(xtest, ytest, ttest, batch_size)

# plots
t_axis = [i for i in range(epochs)]
l_axis = [i for i in range(len(acc2))]

plt.axis([0,  epochs - 0.5, 0, 110])
plt.xticks(t_axis)
plt.plot(t_axis, acc1, color='blue', label='traning_accuracy')
plt.plot(t_axis, loss1, color='blue', label='traning_accuracy')
plt.plot(l_axis, acc2, color='red', label='testing_accuracy')
plt.plot(l_axis, loss2, color='orange', label='testing_loss', linestyle='--')

plt.xlabel("Number of iterations")
plt.ylabel("Percentage")
plt.title("Model Learning")
plt.show()

# single-input test
while True:
    i = int(input(f'index({len(ttest) - 1}):'))
    nn.feedforward(xtest[i])
    print(nn.output(), ttest[i])
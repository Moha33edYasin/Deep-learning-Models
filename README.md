# What is it?

A **neural network framework** implemented entirely from scratch, featuring both a **Multi-Layer Perceptron (MLP)** and a **Convolutional Neural Network (CNN)**.  
  
The project reconstructs fundamental deep learning components, including _convolution kernels, pooling operations, dense-to-flatten transitions, channel handling, optimizers, activation functions, and parameter initialization strategies_.  

The framework emphasizes **modularity, transparency, and user control**, enabling direct inspection and control of the learning mechanics.  

# Why?

To develop a first-principles understanding of neural network behavior rather than relying on high-level libraries.  
Re-implementing core mechanisms provides deeper insight into learning dynamics, gradient flow, and architectural design trade-offs.  

# Next Steps

Extending the framework toward an intelligent agent capable of interpreting human voice commands and mapping them into actions within a 2D environment.  

# Usage

Import the modules:  
```python
from deep_learning.models import *
from deep_learning.activations import *
```

Instantiate a neural network and define the architecture:  

```python
nn = NeuralNetwork(
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    Layer(784),
    Layer(16, ReLU),
    Layer(16, ReLU),
    Layer(10, softmax),
    cost=CCE,  # Categorical Cross-Entropy
    optimizer=Adam(beta1=0.9, beta2=0.99, lr=0.05)
)
```

Train the model:  
```python
loss1, acc1 = nn.learn(xtrain, ytrain, ttrain, lr=0.01, epochs=8, batch_size=325)
```

Evaluate on test data:  
```python
loss2, acc2 = nn.test(xtest, ytest, ttest, batch_size=325)
```

# What is it?

A **neural network framework** implemented entirely from scratch. It reconstructs fundamental deep learning components, including _convolution kernels, pooling operations, dense-to-convolution auto transitions, channel handling, optimizers, activation functions, and parameter initialization strategies_.  

The framework emphasizes **modularity, transparency, and user control**, enabling direct inspection and control of the learning mechanics.  

# Why?

To develop a first-principles understanding of neural network behavior rather than relying on high-level libraries.  
Re-implementing core mechanisms provides deeper insight into learning dynamics, gradient flow, and architectural design trade-offs.  

# Usage

Import the modules:  
```python
from deep_learning.models import *
from deep_learning.methods import *
```

Instantiate a neural network and define the architecture:  

```python
mlp = nn(
        Flatten(),
        Dense(16, ReLU),
        Dense(16, ReLU),
        Dense(10, softmax),
        possible_outcomes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        cost= CCE, # cross categorical entropy
        optimizer= Adam(beta1=0.9, beta2=0.99, lr=0.05)
    )
```

Train the model:  
```python
loss1, acc1 = mlp.learn(xtrain, ytrain, ttrain, epochs=8, batch_size=325)
```

Evaluate on test data:  
```python
loss2, acc2 = mlp.test(xtest, ytest, ttest, batch_size=325)
```
> [!NOTE]
> The above structure achieved ~0.94 testing accuarcy on MNIST.

> [!WARNING]
> Do not use convolutional layers (_Conv_), as they are not fully implemented. 

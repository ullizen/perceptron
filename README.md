Test project for recreating the perceptron architecture.

See reference: https://en.wikipedia.org/wiki/Perceptron

## Architecture
Binary classifier.
Single layer (only one layer of trainable weights).
It's basically a straight line separating the data into two classes.

### Modern definition
Maps a real valued vector (x), to an single binary value (f(x)). 

f(x) = h(w * x + b)

- h is the Heaviside step-function ( > 0 -> 1, < 0 -> 0).
- w * x is the dot product betwen the weight vector and input vector

### Mark I Perceptron
3 layers:
1. Input layer — 400 photocells arranged in a 20 x 20 grid
2. Hidden layer — 512 A-units (association units)
3. Output layer — 8 R-units (response units)

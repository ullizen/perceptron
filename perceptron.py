# Create some data
x = [-1, 1, 2, 3, 4, 5]
y = [1, 1, 1, 0, 0, 0]

# Initial the weights
b = 0
w = 1

# Predict the output
def predict(x):
    preactivation = w * x + b
    if preactivation > 0:
        return 1
    else:
        return 0

def loss(desired, prediction):
    return (desired - prediction)

def learn():
    global w, b, iteration_error
    a = 0.1
    for i in range(len(x)):
        prediction = predict(x[i])
        error = loss(y[i], prediction)
        iteration_error += abs(error) / len(x)
        w = w + a * error * x[i]
        b = b + a * error

# Update until the error is smaller than a specified threshold
threshold = 0.01
max_iterations = 10
iteration_count = 0
iteration_error = 1
while iteration_error > threshold and iteration_count < max_iterations:
    iteration_error = 0
    learn()
    print(f"Iteration {iteration_count}: w: {w}, b: {b}, error: {iteration_error}")
    iteration_count += 1

# Test the model
for i in range(len(x)):
    prediction = predict(x[i])
    print(f"Input: {x[i]}, Prediction: {prediction}, Desired: {y[i]}")
import matplotlib.pyplot as plt

# Create some data
x = [0, 1, 2, 2.5, 4, 5]
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

plt.ion()  # Turn on interactive mode

while iteration_error > threshold and iteration_count < max_iterations:
    iteration_error = 0
    learn()

    # Plot
    plt.clf()
    plt.scatter(x, y, c=y, cmap="bwr", edgecolors="k", label="Data")
    if w != 0:
        boundary_x = -b / w
        plt.axvline(boundary_x, color="red", linestyle="--", label=f"Decision boundary\nx = {-b/w:.2f}")
    plt.ylim(-0.5, 1.5)
    plt.xlim(-1, 6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Epoch {iteration_count}\nError: {iteration_error:.4f}")
    plt.legend()
    plt.grid(True)
    plt.pause(0.5)  # pause for animation effect

    print(f"Iteration {iteration_count}: w: {w}, b: {b}, error: {iteration_error}")
    iteration_count += 1

# Test the model
for i in range(len(x)):
    prediction = predict(x[i])
    print(f"Input: {x[i]}, Prediction: {prediction}, Desired: {y[i]}")

# Plot the data
# fig, ax = plt.subplots()
# ax.plot(x, y)

# # Plot the decision boundary
# if w != 0:
#     boundary_x = -b / w
#     ax.axvline(boundary_x, color="red")

# ax.set_title("Perceptron Decision Boundary")
# plt.show()

plt.ioff()
plt.show()
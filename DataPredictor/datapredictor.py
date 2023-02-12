# Import the necessary modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the model 
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape(1,)),
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer="sgd", loss "mean_squared_error")

# Get the number of values prompted from the user
num_values = int(input("Enter the number of values you want to calculate: "))
input_values = []

# Get the input values prompted from the user
for i in range(num_values):
  input_value = float(input("Enter value {}: " .format(i+1)))
  input_values.append([input_value])
  
input_data = tf.constant(input_values)
predictions = model.predict(input_data)

# Print the prediction
print("Predictions: ", predictions)

x = np.array(input_values).flatten()
y = np.array(predictions).flatten()

# Plot the graph
plt.scatter(x, y, color="blue", label="Prediction")
plt.scatter(x, x, color="red", label="User Input")
plt.xlabel("Input Values")
plt.ylabel("Predictions")
plt.title("Predictions vs Input Values")
plt.legend()
plt.show()

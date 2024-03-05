import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

# Build the neural network model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))  # Input shape is (4,) for Iris dataset
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the trained model weights
model.load_weights('iris_model_weights.h5')  # Assuming the trained model weights are saved in 'iris_model_weights.h5'

# Prompt the user to enter input data
user_input = input("Enter the input data for prediction separated by spaces (sepal length, sepal width, petal length, petal width): ")
user_input = np.array(list(map(float, user_input.split())))  # Convert input string to array of floats

# Make predictions using the trained model
prediction = model.predict(user_input.reshape(1, -1))
predicted_class = np.argmax(prediction)

print("Predicted class:", predicted_class)

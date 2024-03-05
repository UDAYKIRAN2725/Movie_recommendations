import tensorflow as tf
# Here the layers api api used to build the layers in the cnn(convoultional neural network) , model api used to build the model, and the datasets library is used to retrieve the mnist dataset which contains the images of birds, vehicles and others
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
# Here the dataset is divided into two purposes train and test . Initially the image represented as 28x28 array of pixels and each box or pixel has color ranging from 0(black) to 255(white) . so each pixel is divided by 255 which gives outcomes o as black and 1 as white and floating point used as features to the input layer which the model more efficent and more stable. 
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN architecture
# Here the below code build the cnn model which is used to detects the objects . The cnn model have mainly three layers input , hidden and output layers. The hidden layer(identifies edges and corners) again classified into 3 layers which are convolution layer, relu layer and poling layer.The output layer is fully connected layer.
# 1.Here the cnn model is made initially the 28x28x1 input image is made into 3x3x32 which means for every channel(32) 3x3 grid is made 
# 2.above one is forwarded to the poling layer, in this layer max poling is done with 2x2 window size from conv layer which outputs 2x2x32 shape.
# above 2x2x32 image is made into 3x3x64 and then max poling of 2x2 window size
# 5.the output from the previous layer which is 2d in shape is converted into 1d shape
# 6. This line adds a fully connected (dense) layer to the model with 64 units. The ReLU activation function is applied to the output of this layer.
# 7.This line adds the output layer to the model with 10 units, corresponding to the 10 classes in the MNIST dataset (digits 0 through 9)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Reshape input data and add channel dimension for CNN
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

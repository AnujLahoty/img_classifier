# Importing the dataset
import keras
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)


# Currently the labels are stored as 0, 1, 2 so Conversion of labels to one-hot encodings is required.
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)


# We would normalize it but not standardize it.
# In most of CNNs we actually normalize our data but not standardize it.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

'''
SUMMARY TILL NOW:
1 Downloaded the dataset and visualize the images
2 Changed the label to one-hot encodings
3 Scale the image pixel values to take between 0 and 1

'''


# PART 2 : TRAINING
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Building the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()


# Compiling the moedel
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training our model
hist = model.fit(x_train, y_train_one_hot, 
           batch_size=32, epochs=20, 
           validation_split=0.2)

# Visualising our model's loss
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Visualising the accuracy of our model
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Evaluating our model on the 1st test example
model.evaluate(x_test, y_test_one_hot)[1]

model.save('my_cifar10_model.h5')

# Reloading our saved model.
from tensorflow.keras.models import load_model
saved_model = load_model('my_cifar10_model.h5')

# Reading and Resizing the test image according to our trained data.
my_image = plt.imread("test_image.jpg")

# The first thing we have to do is to resize the image of our cat so that we can fit it into our model (input size of 32 * 32 * 3).
from skimage.transform import resize
my_image_resized = resize(my_image, (32,32,3))

img = plt.imshow(my_image_resized)
'''
NOTE:
Note that the resized image has pixel values already scaled between 0 and 1, so we need not apply
the pre-processing steps that we previously did for our training image.
i.e No need for normalizing.When we resized it this step got implemented. 

'''

'''
Below code converts our 3d array to 4d array because model.predict expects a 4-D array instead of a 3-D array
(with the missing dimension being the number of training examples). 
This is consistent with the training set and test set that we had previously.
But for test image there will be only one at a time so.

But if we are providing multiple images at a same time then no need to do the below step.
We can directly use model.predict function.
'''
# Testing our saved model.

import numpy as np
probabilities = model.predict(np.array( [my_image_resized,] ))

# To make code snippet simple to determine and readable we use below code.
number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

'''
NOTE :
argsort would sort in ascending order with the highest probabilities at the last.
'''

index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])




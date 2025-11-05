### Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import os
from os import listdir
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import img_to_array, array_to_img
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.utils import to_categorical

print(tf.__version__)

### Defining the path of dataset directory
dataset_path = "D:/Projects/Major Project/Plant-Disease-Detection/Dataset"

### Visualizing the images and Resize images
# Checking th Dataset (Plotting 12 images to check dataset)
plt.figure(figsize = (12, 12))
dataset_path_sample = "D:/Projects/Major Project/Plant-Disease-Detection/Dataset/Corn_(maize)___healthy"
for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(dataset_path_sample + '/' + random.choice(sorted(os.listdir(dataset_path_sample))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10) # width of image
    plt.ylabel(rand_img.shape[0], fontsize = 10) # height of image

### Convert the images into a Numpy array and normalize them
# Converting Images to array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (128, 128))
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

labels = os.listdir(dataset_path)
print(labels)
root_dir = listdir(dataset_path)

image_list, label_list = [], []
all_labels = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Soyabean_Septoria_Brown_Spot', 'Soyabean_Vein Necrosis', 'Soybean___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight']
binary_labels = [i for i in range(0, len(labels))]

# Reading and converting image to numpy array
for temp, directory in enumerate(root_dir): 
    plant_image_list = listdir(f"{dataset_path}/{directory}")
    print(directory)
    for files in plant_image_list:
        image_path = f"{dataset_path}/{directory}/{files}"
        img_array = convert_image_to_array(image_path)
        if img_array is not None and img_array.size != 0:
            image_list.append(img_array)
            label_list.append(binary_labels[temp])

# Convert image_list and label_list to numpy arrays BEFORE splitting
image_array = np.array(image_list, dtype=np.float16) / 255.0
label_array = np.array(label_list)

# Reshape image_array to proper shape
image_array = image_array.reshape(-1, 128, 128, 3)

# Splitting the dataset into train, validate and test sets
x_train, x_test, y_train, y_test = train_test_split(image_array, label_array, test_size=0.2, random_state=10)

# Performing one-hot encoding on target variable AFTER splitting
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Splitting the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)

### Creating the model architecture, compile the model and then fit it using the training data
model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), padding = "same", input_shape = (128, 128, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Conv2D(16, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(8, activation = "relu"))
model.add(Dense(15, activation = "softmax"))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001), metrics = ['accuracy'])

# Training the model
epochs = 50
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

# Save the model
model.save("D:/Projects/Major Project/Plant-Disease-Detection/Model/plant_disease_model.h5")

### Plot the accuracy and loss against each epoch
plt.figure(figsize = (12, 5))
plt.plot(history.history['accuracy'], color = 'r')
plt.plot(history.history['val_accuracy'], color = 'b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

print("Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1] * 100}")

### Make predictions on testing data
y_pred = model.predict(x_test)

### Visualizing the original and predicted labels for the test images
img = array_to_img(x_test[11])
img
print("Originally : ", all_labels[np.argmax(y_test[11])])
print("Predicted : ", all_labels[np.argmax(y_pred[11])])  # Fixed index(11)
print(y_pred[2])
for i in range(50):
    print(all_labels[np.argmax(y_test[i])], " ", all_labels[np.argmax(y_pred[i])])

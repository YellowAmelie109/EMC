#Fix the fit data sets
#Prevent overfitting
#Change process to own process 
#Globals

# Uses TensorFlow to build and train a CNN on the CIFAR-10 dataset

import datetime
import matplotlib.pyplot as plt
#import PIL

from tensorflow import keras
from keras import datasets, layers
from keras.utils import to_categorical

from keras.models import Sequential

import testModel

numClasses = 10
inputShape=(32,32,3)

#Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0



#Augment the images to prevent overfitting
data_augmentation = Sequential([
    layers.RandomFlip("horizontal",
                      input_shape=inputShape),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

#Define the model with 3 convolutional layers with relu activation and 3 max pooling layers
model = Sequential()

model.add(data_augmentation)

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))



model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs=50
history = model.fit(train_images, train_labels, epochs=epochs, 
                    validation_data=(test_images, test_labels))


#Save the model
date = datetime.datetime.now()
modelname='YAY'+date.strftime("%d")+date.strftime("%b")+'.keras'
model.save(modelname)

#Plot the accuracy and loss as epochs increase as a line graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

testModel.mainTest(modelname)
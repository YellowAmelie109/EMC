#Fix the fit data sets
#Prevent overfitting
#Change process to own process 
#Globals

# Uses TensorFlow to build and train a CNN on the CIFAR-10 dataset
import time

import matplotlib.pyplot as plt
#import numpy as np
#import PIL

import tensorflow as tf

from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds

start=time.time()

train_ds,ds_meta = tfds.load('cifar10', split='train', with_info=True, as_supervised=True)
val_ds = tfds.load('cifar10', split='test', as_supervised=True)
assert isinstance(train_ds, tf.data.Dataset)
assert isinstance(val_ds,tf.data.Dataset)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



#Define the model with 3 convolutional layers with relu activation and 3 max pooling layers
num_classes = ds_meta.features["label"].num_classes
inputShape=ds_meta.features["image"].shape

model = keras.models.load_model("higherDropout2Oct.keras")
model.summary()

#Train the model
epochs=1
history = model.fit(
  train_ds.batch(32),
  validation_data=val_ds.batch(32),
  epochs=epochs,
  verbose=2
)

model.save('higherDropout2Oct.keras')

print((time.time()-start)/60)

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
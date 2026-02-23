import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import datetime

import testModel

# Load CIFAR-10 dataset
(train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

# Normalize pixel values to [0, 1]
train_data = train_data / 255.0
val_data = val_data / 255.0

# One-hot encode labels
train_labels = to_categorical(train_labels, num_classes=10)
val_labels = to_categorical(val_labels, num_classes=10)

#Augment that training data to prevent overfitting
augmentation_layers = [
    layers.RandomFlip("horizontal",
                      input_shape=(32,32,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x


train_data = data_augmentation(train_data)

# Build the model using VGG16 as base

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

inputs = base_model.input


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(10, activation='softmax')(x)
pretrain_model = Model(inputs=inputs, outputs=predictions)

# Freeze the layers of the pretrained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the pretraining model
pretrain_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the pretraining model
history_pretrain = pretrain_model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))

# Fine-tuning the model
for layer in pretrain_model.layers[-4:]:
    layer.trainable = True

# Compile the model with a lower learning rate
pretrain_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the fine-tuning model
history_finetune = pretrain_model.fit(train_data, train_labels, epochs=5, validation_data=(val_data, val_labels))

date = datetime.datetime.now()
modelname='Geek'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
pretrain_model.save(modelname)


# Evaluate the model on validation data
val_loss, val_accuracy = pretrain_model.evaluate(val_data, val_labels, verbose=2)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Validation Loss: {val_loss:.4f}')

# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history_pretrain.history['accuracy'] + history_finetune.history['accuracy'])
plt.plot(history_pretrain.history['val_accuracy'] + history_finetune.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_pretrain.history['loss'] + history_finetune.history['loss'])
plt.plot(history_pretrain.history['val_loss'] + history_finetune.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

testModel.mainTest(modelname)
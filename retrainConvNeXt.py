import matplotlib.pylab as plt

import datetime
import keras
from keras.utils import to_categorical

import testModel

numClasses = 10
inputShape=(32,32,3)

#Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

#train_images = train_images / 255.0
#test_images = test_images / 255.0


#train_labels = to_categorical(train_labels, num_classes=10)
#test_labels = to_categorical(test_labels, num_classes=10)

augmentation_layers = [
    keras.layers.RandomFlip("horizontal",
                      input_shape=inputShape),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
  ]

'''def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x


x_train = x_train.map(lambda x, y: (data_augmentation(x), y))'''

base_model = keras.applications.ConvNeXtBase(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(32, 32, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.



# Create new model on top
inputs = keras.Input(shape=(32, 32, 3))

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(10)(x)
model = keras.Model(inputs, outputs)

# Freeze the base_model
for layer in base_model.layers:
    layer.trainable = False

model.summary(show_trainable=True)

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


epochs=5
history = model.fit(train_images, train_labels, epochs=epochs, 
                    validation_data=(test_images, test_labels))

# Fine-tuning the model
for layer in model.layers[-4:]:
    layer.trainable = True

# Compile the model with a lower learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the fine-tuning model
history_finetune = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))


#predicted_batch = model.predict()
#predicted_id = tf.math.argmax(predicted_batch, axis=-1)
#predicted_label_batch = class_names[predicted_id]
#print(predicted_label_batch)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

#for n in range(30):
#  plt.subplot(6,5,n+1)
#  plt.imshow(image_batch[n])
#  plt.title(predicted_label_batch[n].title())
#  plt.axis('off')
#_ = plt.suptitle("Model predictions")


date = datetime.datetime.now()
modelname='Inception'+date.strftime("%d")+date.strftime("%b")
model.save(modelname)

testModel.mainTest(modelname)

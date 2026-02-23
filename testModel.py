import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def runTest(model,path,height=32,width=32):

    img_height=height
    img_width=width

    img = keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array,verbose=0)
    score = tf.nn.softmax(predictions[0])

    label_names=unpickle("cifar-10-python/cifar-10-batches-py/batches.meta")[b'label_names']

    print(
        "most likely belongs to category '{}' with a {:.2f}% confidence."
        .format(str(label_names[np.argmax(score)])[2:-1], 100 * np.max(score))
    )

def mainTest(model):
    model=keras.models.load_model(model)

    dir="C:/Users/ameli/Documents/EMC/Images"
    for name in os.listdir(dir):
        print(f"Image '{name.split('.')[0]}'",end=' ')
        path = dir+"/"+name
        runTest(model,path)
        print()

if __name__ == "__main__":
    mainTest("Geek20260220-231316")
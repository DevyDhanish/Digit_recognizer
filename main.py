import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("3.png").convert("L")
img_resized = img.resize((28,28))
img_array = np.array(img_resized)

model = tf.keras.models.load_model("digit.h5")

classes = {
    1:"One",
    2:"Two",
    3:"Three",
    4:"Four",
    5:"Five",
    6:"Six",
    7:"Seven",
    8:"Eight",
    9:"Nine",
    0:"Zero"
}

def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    image /= 255
    return image

test = normalize(img_array)

img = np.array(test)
img_flat = img.reshape(1,28,28,1)
title = model.predict(img_flat)

title = classes[np.argmax(title)]

plt.imshow(img, cmap="gray")
plt.title(f"I think it is a {title}")
plt.show()
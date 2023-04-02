import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

# img = X_train[5]
# plt.imshow(img, cmap="gray")
# plt.title(f"{Y_train[5]}")
# plt.show()

def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    image /= 255
    return image

X_train = normalize(X_train)
X_test = normalize(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(X_train,Y_train, epochs=20, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, Y_test)

print(f"Model accuracy is {test_acc}")
model.save("digit.h5")
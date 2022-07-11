
import tensorflow as tf
from tensorflow import keras
from mlxtend.data import loadlocal_mnist

# Read the data
X, y = loadlocal_mnist(
        images_path='../RAW_data/IDX/images.idx3-ubyte',
        labels_path='../RAW_data/IDX/labels.idx3-ubyte')
#
print(X.shape)
X = X/255.0

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(50,  activation='relu'),
    keras.layers.Dense(50,  activation='relu'),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X, y, epochs=5)
model.save('../RAW_data/IDX/model.h5')




import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
all_data = [0,1,2,3,4,5,6]

CIFAR_DIR = 'cifar-10-batches-py/'

for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
data_batch_1 = all_data[1]
data_batch_2 = all_data[2]
data_batch_3 = all_data[3]
data_batch_4 = all_data[4]
data_batch_5 = all_data[5]
test_batch = all_data[6]

X = data_batch_1[b"data"]

#Break up the data into 10000 images, 32x32 image with 3 color
#Also the data was encoded oddly, as 3 is usually at the end, so transpose will have it be 32x32x3
#astype
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

plt.imshow(X[4])

plt.show()

def one_hot_encode(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarHelper():

    def __init__(self):
        self.i = 0

        self.all_train_batches = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):

        print("Setting up Training Images and Labels")

        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0,2,3,1)/255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting up Test Image and Labels")

        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)


ch = CifarHelper()
ch.set_up_images()

model = tf.keras.Sequential([
    # The Input layer expects images of shape 32x32 with 3 channels.
    tf.keras.layers.Input(shape=(32, 32, 3)),

    # First convolutional block: Conv -> ReLU -> MaxPooling
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten the feature maps and pass through Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # Output layer for 10 classes (CIFAR-10)
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()  # Print a summary of your model architecture

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    ch.training_images,
    ch.training_labels,
    epochs=10,
    batch_size=100,
    validation_data=(ch.test_images, ch.test_labels)
)

test_loss, test_accuracy = model.evaluate(ch.test_images, ch.test_labels)
print(f"Test accuracy: {test_accuracy:.2f}")

model.save('my_cifar_model.keras')






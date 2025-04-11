import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

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

X = test_batch[b"data"]
Y = test_batch[b"labels"]




#Break up the data into 10000 images, 32x32 image with 3 color
#Also the data was encoded oddly, as 3 is usually at the end, so transpose will have it be 32x32x3
#astype
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")




label_names = batch_meta[b'label_names']
label_names = [name.decode('utf-8') for name in label_names]
print("CIFAR-10 label names:", label_names)

model = tf.keras.models.load_model('my_cifar_model.keras')


for i in range(5):
    img_selector = random.randint(0, 9999)

    plt.imshow(X[img_selector])

    plt.show()

    single_image = X[img_selector].astype('float32') / 255.0

    predictions = model.predict(np.expand_dims(single_image, axis=0))

    predicted_classes = np.argmax(predictions, axis=1)

    # Print both the numerical label and its corresponding name
    print('Actual label number:', Y[img_selector])
    print("Actual label name:", label_names[Y[img_selector]])

    print("Predicted label number:", predicted_classes)
    print("Predicted label name:", label_names[predicted_classes[0]])
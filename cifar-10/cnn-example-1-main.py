import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pickle
import process_data
import model_utils

# guide:
# https://colab.research.google.com/drive/1P6JMJuwrrh_Ht6xohRz-EpJai9KuWSaN#scrollTo=aPqeddhcPwpc
# dataset: https://www.cs.toronto.edu/~kriz/cifar.html
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz


def plot_image(image_index, image_array, image_labels, label_names):
    plt.imshow(image_array[image_index], cmap=plt.cm.binary)
    a = image_labels[image_index]
    label_word = label_names[a[0]]
    plt.xlabel(label_word)
    plt.show()


def save_history(file_name, model_history):
    pickle.dump(model_history, open(f"cifar-10/saved_histories/{file_name}.pkl", "wb"))


def load_history(file_name):
    return process_data.unpickle(f"cifar-10/saved_histories/{file_name}.pkl")


def save_model(file_name, model):
    pickle.dump(model, open(f"cifar-10/saved_models/{file_name}.pkl", "wb"))


def load_model(file_name):
    return process_data.unpickle(f"cifar-10/saved_models/{file_name}.pkl")


# uncomment to reprocess from raw data
# process_data.process_all_data()


# load processed data
# (
#     train_images,
#     train_labels,
#     test_images,
#     test_labels,
#     label_names,
# ) = process_data.load_all_data()


# ~~~~ load data in a different way:
# for some reason thsi leads to MUCH higher accuracy than manually loading
# the data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# not sure why... probably a wrong manual transformation on my part
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

label_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ~~~~

# plot an image:
# plot_image(25000, train_images, train_labels, label_names)


# build model
model = model_utils.generate_model()

# run the model and save the history
epochs = 4
run = 17
file_name = f"run{run}-{epochs}epochs"
saved_model_path = f"cifar-10/saved_models/{file_name}.h5"

# train and save
# history = model_utils.run_model(
#     model, epochs, train_images, train_labels, test_images, test_labels
# )
# save_history(file_name, history.history)
# model.save_weights(saved_model_path)


# load saved
loaded_history = load_history(file_name)
loaded_model = model_utils.generate_model()
loaded_model.load_weights(saved_model_path)
loaded_model.summary()

print("loaded:")
test_loss, test_acc = loaded_model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)

import numpy as np
import pickle

#  LOAD AND SPLIT DATASET
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


def convert_img_to_std_format(raw_image_data):
    red = raw_image_data[0:1024].reshape(32, 32)
    blue = raw_image_data[1024:2048].reshape(32, 32)
    green = raw_image_data[2048:].reshape(32, 32)
    image = np.dstack((red, blue, green))

    image = image / 255

    return image


def convert_image_list_to_std_format(image_list):
    for index, image in enumerate(image_list):
        image_list[index] = convert_img_to_std_format(image)


# a shorthand method for loading all raw data from downloaded raw data
# converting it to a different format for plotting, then pickling it
# for later use. it returns test
# train_images, train_labels, test_images, test_labels, label_names
def process_all_data():
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

    # ~~~~~~~~~~~~ process images ~~~~~~~~~~~~

    # group batches into 1 batch
    batch1 = unpickle("cifar-10/cnn-example-1-data/raw_data/data_batch_1")
    batch2 = unpickle("cifar-10/cnn-example-1-data/raw_data/data_batch_1")
    batch3 = unpickle("cifar-10/cnn-example-1-data/raw_data/data_batch_1")
    batch4 = unpickle("cifar-10/cnn-example-1-data/raw_data/data_batch_1")
    batch5 = unpickle("cifar-10/cnn-example-1-data/raw_data/data_batch_1")
    batch_test = unpickle("cifar-10/cnn-example-1-data/raw_data/test_batch")

    batches = [batch1, batch2, batch3, batch4, batch5]

    # populate test/train image and label arrays
    train_images = []
    train_labels = []
    for batch in batches:
        for i in range(10000):
            train_images.append(batch["data"][i])
            train_labels.append([batch["labels"][i]])

    test_images = []
    test_labels = []
    for i in range(10000):
        test_images.append(batch_test["data"][i])
        test_labels.append([batch_test["labels"][i]])

    # convert images to standard format for viewing
    convert_image_list_to_std_format(train_images)
    convert_image_list_to_std_format(test_images)

    # convert to numpy array
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)
    label_names = np.asarray(label_names)

    # pickle the arrays so we don't have to reprocess images
    pickle.dump(
        train_images, open("cifar-10/cnn-example-1-data/train_images.pkl", "wb")
    )
    pickle.dump(
        train_labels, open("cifar-10/cnn-example-1-data/train_labels.pkl", "wb")
    )
    pickle.dump(test_images, open("cifar-10/cnn-example-1-data/test_images.pkl", "wb"))
    pickle.dump(test_labels, open("cifar-10/cnn-example-1-data/test_labels.pkl", "wb"))
    pickle.dump(label_names, open("cifar-10/cnn-example-1-data/label_names.pkl", "wb"))


def load_all_data():
    train_images = unpickle("cifar-10/cnn-example-1-data/train_images.pkl")
    train_labels = unpickle("cifar-10/cnn-example-1-data/train_labels.pkl")
    test_images = unpickle("cifar-10/cnn-example-1-data/test_images.pkl")
    test_labels = unpickle("cifar-10/cnn-example-1-data/test_labels.pkl")
    label_names = unpickle("cifar-10/cnn-example-1-data/label_names.pkl")

    return train_images, train_labels, test_images, test_labels, label_names

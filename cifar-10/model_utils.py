import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def generate_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # print(model.summary())

    # top?
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))
    # print(model.summary())

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def run_model(model, epochs, train_images, train_labels, test_images, test_labels):

    # train the model (long time)
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
    )

    return history

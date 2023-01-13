import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import layers
from keras import Sequential
import pathlib
import random

def training_loop(opt):
    neuron1 = random.randint(64, 128)
    neuron2 = random.randint(32, 64)
    neuron3 = random.randint(16, 32)
    dropout1 = (random.randint(0, 6)) / 10
    dropout2 = (random.randint(0, 6)) / 10
    dropout3 = (random.randint(0, 6)) / 10
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(neuron1, activation='relu', name='layer1'),
        layers.Dropout(dropout1),
        layers.Dense(neuron2, activation='relu', name='layer2'),
        layers.Dropout(dropout2),
        layers.Dense(neuron3, activation='relu', name='layer3'),
        layers.Dropout(dropout3),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save('model'+ str(nazwa))




    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    

    # Save accuracy values to a file
    np.savetxt("model"+ str(nazwa)+"/acc.txt", acc, delimiter=",", fmt='%.2f')
    np.savetxt("model"+ str(nazwa)+"/val_acc.txt", val_acc, delimiter=",", fmt='%.2f')

    # Save loss values to a file
    np.savetxt("model"+ str(nazwa)+"/loss.txt", loss, delimiter=",", fmt='%.2f')
    np.savetxt("model"+ str(nazwa)+"/val_loss.txt", val_loss, delimiter=",", fmt='%.2f')

if __name__ == '__main__':
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    num_classes = len(class_names)

    optimizers = ['rmsprop', 'sgd', 'adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam']
    nazwa = 1
    for i in range(10):
        for j in optimizers:
            training_loop(j)
            nazwa += 1

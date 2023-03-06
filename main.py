import matplotlib.pyplot as plt
import tensorflow as tf
import json
from keras import layers
import keras_tuner as kt
import pathlib
from hyperModel import create_HyperModel


def train_and_save_model(img_height, img_width, num_classes, epochs, train_ds, val_ds, optimizer):
    model = create_HyperModel(img_height, img_width, num_classes)  # tu nalezy wymienic model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("config_results")
    model.summary()
    return model, history


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

    # model creation

    img_height = 180
    img_width = 180
    num_classes = 5
    epochs = 10

    model, history = train_and_save_model(img_height, img_width, num_classes, epochs, train_ds, val_ds,
                                          optimizer='adam')
    tuner = kt.Hyperband(create_HyperModel,
                         objective='val_accuracy',
                         max_epochs=epochs,
                         factor=3,
                         directory='auto_trening',
                         project_name='aiir')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_ds, epochs=50, validation_data=val_ds, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, epochs=50, validation_data=val_ds)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(train_ds, epochs=best_epoch, validation_data=val_ds)

    eval_result = hypermodel.evaluate(val_ds)
    print("[test loss, test accuracy]:", eval_result)

    with open(f'models/hyper_model - {epochs}.txt', 'a') as f:
        json.dump({'params': model.get_config(), 'history': history.history}, f)

    #  Here model creation has ended

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

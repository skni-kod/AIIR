import tensorflow as tf
from tensorflow import keras
from keras import layers
import pathlib
import keras_tuner as kt
def model_builder(hp):
  model = keras.Sequential()


  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-256
  hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
  hp_units2 = hp.Int('units2', min_value=16, max_value=128, step=16)
  hp_units3 = hp.Int('units3', min_value=8, max_value=64, step=8)

  # Image processing
  model.add(keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)))
  model.add(keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu'))
  model.add(keras.layers.MaxPooling2D())
  model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
  model.add(keras.layers.MaxPooling2D())
  model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
  model.add(keras.layers.MaxPooling2D())
  model.add(keras.layers.Flatten(input_shape=(180, 180, 3)))

  # Image identification
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(units=hp_units3, activation='relu'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(num_classes))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

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

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='auto_training',
                         project_name='aiir')


    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_ds, epochs=50, validation_data=val_ds, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data for 20 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, epochs=20, validation_data=val_ds)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(train_ds, epochs=best_epoch, validation_data = val_ds)



    eval_result = hypermodel.evaluate(val_ds)
    print("[test loss, test accuracy]:", eval_result)

    hypermodel.save("hipermodel_ex")
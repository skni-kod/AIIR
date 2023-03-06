from tensorflow import keras
from kerastuner import HyperParameters


def create_HyperModel(img_height, img_width, num_classes):
    hyperModel = keras.Sequential()
    hp = HyperParameters()
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_units2 = hp.Int('units2', min_value=16, max_value=256, step=16)
    hp_units3 = hp.Int('units3', min_value=8, max_value=128, step=8)

    hyperModel.add(keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu'))
    hyperModel.add(keras.layers.MaxPooling2D())
    hyperModel.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    hyperModel.add(keras.layers.MaxPooling2D())
    hyperModel.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    hyperModel.add(keras.layers.MaxPooling2D())

    hyperModel.add(keras.layers.Flatten(input_shape=(img_height, img_width, 3)))

    hyperModel.add(keras.layers.Dense(units=hp_units, activation='relu'))
    hyperModel.add(keras.layers.Dropout(0.2))
    hyperModel.add(keras.layers.Dense(units=hp_units2, activation='relu'))
    hyperModel.add(keras.layers.Dropout(0.2))
    hyperModel.add(keras.layers.Dense(units=hp_units3, activation='relu'))
    hyperModel.add(keras.layers.Dropout(0.2))
    hyperModel.add(keras.layers.Dense(units=num_classes))  # bo 5 roznych kwaitow to 5 roznych neuronów

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    return hyperModel

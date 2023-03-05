from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# additional libs (might change later)
import numpy as np
import matplotlib.pyplot as plt

def image_prediction(file_with_model,img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    model = load_model(file_with_model)
    prediction = model.predict(img_preprocessed)
    for array in prediction:
        print("predictions: ",array)
        print("predicted category:",end=" ")
        if array[0] == max(array):
            print("daisy")
        elif array[1] == max(array):
            print("dandelion")
        elif array[2] == max(array):
            print("rose")
        elif array[3] == max(array):
            print("sunfower")
        else:
            print("tulip")
    # optional code (displays rescaled image)
    # plt.imshow(img)
    # plt.show()
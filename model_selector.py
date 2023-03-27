import json
import tensorflow as tf
from tensorflow import keras


if __name__ == '__main__':
    with open('training_results/results.json', 'r') as input_file:
        data = input_file.read()
        results1 = json.loads(data)
        results1.pop('model_24')
    with open('training_results/results2.json', 'r') as input_file:
        data = input_file.read()
        results2 = json.loads(data)

    results = results1 | results2
    finals = dict()
    for model, params in results.items():
        finals[model] = params['train_accuracy'] + params['val_accuracy'] * 2 + (1 - params['test_loss']) * 4 + params['test_accuracy'] * 5

    finals = dict(sorted(finals.items(), key=lambda x: x[1], reverse=True))

    top = 0
    for model, result in finals.items():
        print(f"Model: {model}, result: {result}")
        top += 1
        if top == 3:
            break

    model = keras.models.load_model('training_results/model_45')
    print(model.layers)
    keras.models.save_model(model, 'training_results/model_45_light', save_format="h5")

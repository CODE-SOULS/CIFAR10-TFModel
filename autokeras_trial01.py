import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Initialize the image classifier.
clf = ak.ImageClassifier(
    overwrite=False,
    max_trials=1)
# Feed the image classifier with training data.
clf.fit(x_train, y_train,validation_split=0.20, epochs=2, batch_size=8)

# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
model = clf.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

try:
    model.save("model_cifar_autokeras", save_format="tf")
except:
    model.save("model_cifar_autokeras.h5")
#
# # from tensorflow.keras.models import load_model
# # import autokeras as ak
# # loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
# # loaded_model.summary()
#
# import numpy as np
# import pickle
# import os
#
# # data_dir = r"E:\Courses TAMU\Fall 2020\CSCE 636\Final Project\starter_code\data\cifar-10-batches-py"
# # training_files = [
# #     os.path.join(data_dir, 'data_batch_%d' % i)
# #     for i in range(1, 6)
# # ]
# # x_train = []
# # for training_file in training_files:
# #     with open(training_file, 'rb') as f:
# #         d = pickle.load(f, encoding='bytes')
# #     x_train.append(d[b'data'].astype(np.float32))
# # x_train = np.concatenate(x_train, axis=0)
# # print(x_train.shape)
#
# from tensorflow.python.keras.models import model_from_json
# import autokeras as ak
# # path = r"E:\Courses TAMU\Fall 2020\CSCE 636\Final Project\image_classifier\trial_f4ae6841ee03492bdb0c530b1c24ee89\trial.json"
# # json_file = open(path, 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# #
# # loaded_model=model_from_json(loaded_model_json, custom_objects=ak.CUSTOM_OBJECTS)
# # print(loaded_model)
#
# ak.auto_model("f4ae6841ee03492bdb0c530b1c24ee89")
from tensorflow.keras.datasets import cifar10
import autokeras as ak

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# # Initialize the image classifier.
# clf = ak.ImageClassifier(overwrite=False, max_trials=1)
# # Feed the image classifier with training data.
# clf.fit(x_train, y_train, validation_split=0.20, epochs=2, batch_size=8)
# # Predict with the best model.
# predicted_y = clf.predict(x_test)
# print(predicted_y)
# # Evaluate the best model with testing data.
# print(clf.evaluate(x_test, y_test))
# model = clf.export_model()
# try:
#     model.save("model_cifar_autokeras", save_format="tf")
# except:
#     model.save("model_cifar_autokeras.h5")

import tensorflow as tf
import autokeras as ak

# new_model = tf.keras.models.load_model(r'/home/haruiz/Workspace/CIFAR10-TFModel/image_classifier/trial_8e8665c76e48c974e729d6aeacc33fdf/checkpoints/epoch_0/checkpoint')
# checkpoint = tf.train.load_checkpoint(r'/home/haruiz/Workspace/CIFAR10-TFModel/image_classifier/trial_8e8665c76e48c974e729d6aeacc33fdf/checkpoints/epoch_0/checkpoint')
# clf = ak.ImageClassifier(overwrite=False, max_trials=1)
clf = ak.ImageClassifier(overwrite=False, max_trials=1)
model = clf.tuner.get_best_model()
print(model)

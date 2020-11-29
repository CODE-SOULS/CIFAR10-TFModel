### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs


parser = argparse.ArgumentParser()
parser.add_argument("mode", default="train", help="train, test or predict", nargs="?")
parser.add_argument("data_dir", default="./../data", help="path to the data", nargs="?")
parser.add_argument(
    "--save_dir", default="./../output", help="path to save the results", nargs="?"
)
args = parser.parse_args()

if __name__ == "__main__":
    model = MyModel(model_configs)
    if args.mode == "train":
        x_train, y_train, x_test, y_test = load_data(args.data_dir)
        #x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
        #model.train(x_train, y_train, training_configs, x_valid, y_valid)

        model.train(x_train, y_train, training_configs, x_test, y_test)
        model.evaluate(x_test, y_test)
    elif args.mode == "test":
        # Testing on public testing dataset
        _, _, x_test, y_test = load_data(args.data_dir)
        model.evaluate(x_test, y_test)
    elif args.mode == "predict":
        # Predicting and storing results on private testing dataset
        x_test = load_testing_images(args.data_dir)
        predictions = model.predict_prob(x_test)
        np.save(os.path.join(args.save_dir, "predictions"), predictions)
### END CODE HERE

import argparse
from datetime import datetime
import json
import multiprocessing
import os

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from keras.callbacks import TensorBoard

from data import *
from model import MainModel
from predict_direct import predict
from eval_direct import evaluate_predictions


def train_eval_model(model_spec):
    lang, size, model_type, batch_size, short_strings = model_spec

    train_data = os.path.join("data_gen", size, f"{lang}_Train.txt")
    val_data = os.path.join("data_gen", size, f"{lang}_Dev.txt")
    model_dir = "models" if not short_strings else "models_short"
    model_path = os.path.join(model_dir, f"Uni_{model_type}_NoDrop_{lang}_{size}")
    os.makedirs(model_path, exist_ok=True)

    # set up hyperparamters
    model_types = ["simple", "gru", "lstm", "transformer"]
    hparams = {
        "dropout": [0.1, 0.1, 0.0, 0.1],
        "embed_dim": [32, 32, 256, 256],
        "epochs": 4 * [64],
        "learning_rate": [0.0001, 0.01, 0.0001, 0.0001],
        "loss": 4 * ["BinaryCrossEntropy"],
        "num_ff_layers": [4, 2, 2, 2],
        "optimizer": ["Adam", "RMSprop", "RMSprop", "Adam"],
    }
    hparams = {
        m_type: {hparam: hparams[hparam][idx] for hparam in hparams}
        for idx, m_type in enumerate(model_types)
    }

    dropout = hparams[model_type]["dropout"]
    embed_dim = hparams[model_type]["embed_dim"]
    epochs = hparams[model_type]["epochs"]
    learning_rate = hparams[model_type]["learning_rate"]
    loss = hparams[model_type]["loss"]
    num_ff_layers = hparams[model_type]["num_ff_layers"]
    optimizer = hparams[model_type]["optimizer"]

    losses = {
        "BinaryCrossEntropy": keras.losses.BinaryCrossentropy(),
        "MeanSquaredError": keras.losses.MeanSquaredError(),
    }
    optimizers = {
        "RMSprop": keras.optimizers.RMSprop,
        "Adam": keras.optimizers.Adam,
        "SGD": keras.optimizers.SGD,
    }
    loss = losses[loss]
    optimizer = optimizers[optimizer](learning_rate=learning_rate)

    vocabulary = {"pad": 0}
    vocabulary, x_train, y_train = parse_dataset(train_data, vocabulary)
    vocabulary, x_val, y_val = parse_dataset(val_data, vocabulary)
    vocab_file = os.path.join(model_path, "vocab.txt")

    max_length = 64
    tf.random.set_seed(1234)

    x_train = tf.constant(pad_data(x_train, vocabulary))
    x_val = tf.constant(pad_data(x_val, vocabulary))
    y_train = tf.constant(y_train)
    y_val = tf.constant(y_val)

    if short_strings:
        short_string_dir = "../tmp/shortgen/OnlyShort"
        short_string_file = os.path.join(short_string_dir, f"{lang}_TrainOS.mlrt")
        vocabulary, x_short, y_short = parse_dataset(short_string_file, vocabulary)

        x_short = tf.constant(pad_data(x_short, vocabulary))
        y_short = tf.constant(y_short)
        x_train = tf.concat([x_short, x_train], axis=0)
        y_train = tf.concat([y_short, y_train], axis=0)

    save_vocab(vocab_file, vocabulary)

    config = {
        "model_type": model_type,
        "dropout": dropout,
        "embed_dim": embed_dim,
        "vocab_size": len(vocabulary),
        "ff_dim": 64,
        "num_ff_layers": num_ff_layers,
        "bidi": False,
    }
    model = MainModel(**config)

    config_file = f"{model_path}/config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanAbsoluteError(),
        ],
    )

    checkpoint_path = os.path.join(model_path, "checkpoint.ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, monitor="val_acc", mode="max"
    )

    time_now = datetime.now()
    time_now_format = time_now.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_path}/logs/fit/{time_now_format}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [checkpoint_callback, tensorboard_callback]

    training_record = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[],
    )

    total_time = (datetime.now() - time_now).total_seconds()
    print(f"TOTAL TRAINING TIME IN SECONDS: {total_time}\n")

    model.summary()

    test_size = "Large"
    test_types = ["SR", "SA", "LR", "LA"]
    data_files = [
        os.path.join("data_gen", test_size, f"{lang}_Test{test_type}.txt")
        for test_type in test_types
    ]
    for test_type, data_file in zip(test_types, data_files):
        predict(model, model_path, data_file, f"Test{test_type}", vocabulary)
        evaluate_predictions(f"{model_path}/Test{test_type}_pred.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, required=True)
    model_file = parser.parse_args().model_file

    batch_size = 64
    short_strings = False

    model_specs = [
        line.split() + [batch_size, short_strings]
        for line in open(model_file).readlines()
    ]

    time_now = datetime.now()

    num_processes = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(train_eval_model, model_specs)

    total_time = (datetime.now() - time_now).total_seconds()
    print(f"TOTAL TIME IN SECONDS: {total_time}\n")


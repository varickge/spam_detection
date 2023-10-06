import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from keras.layers import *
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import AutoTokenizer, BertModel


def fc_model_softmax(input_num=768):
    input_ = tf.keras.Input(shape=(input_num,))
    x = Dense(512, activation="relu", kernel_initializer="he_normal")(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    pred = Dense(2, activation="softmax")(x)

    model = Model(input_, pred)
    return model


def lr_exp_decay(epoch, initial_lr=0.003, decay_rate=0.1):
    return initial_lr * decay_rate**epoch


def trainer(model, data, weights_path, batch_size=32, epochs=30, learning_rate=0.003):
    X_train, X_val, y_train, y_val = data

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            decay=0,
            amsgrad=False,
        ),
    )
    model.load_weights(weights_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        weights_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    #     schedule = tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)
    callbacks_list = [checkpoint]

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(X_val, y_val),
    )

    return history


def get_metrics(model, weights_path, X_test, y_test):
    model.load_weights(weights_path)

    y_pred = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred, axis=-1)

    tp = ((y_pred == 1) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()

    return tp, tn, fp, fn


def create_embeddings(orig_data, model, batch_size=16):
    num_samples = len(orig_data)
    # Calculate the number of batches
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Load the BERT models
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    x_list = []
    for i in range(num_batches):
        # Split data
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_texts = orig_data[start_idx:end_idx]

        # Tokenize the batch_texts and padding to the
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, max_length=1024
        )

        # Disable gradient calculation during inference.
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the embeddings
        x_batch = outputs.last_hidden_state[:, 0, :]
        x_list.append(x_batch)

    x = torch.cat(x_list, dim=0)

    return x


def create_embeddings_for_inference(data, model):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(data, return_tensors="pt", padding=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
    x = outputs.last_hidden_state[:, 0, :]

    return x


def to_tf(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

import os

import tensorflow as tf
from transformers import AutoTokenizer, BertModel, logging

from utils import create_embeddings_for_inference, fc_model_softmax

logging.set_verbosity_error()


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    embedding_model = BertModel.from_pretrained("bert-base-uncased")

    fc_model = fc_model_softmax()
    fc_model.load_weights("model/model.hdf5")

    try:
        while True:
            user_input = input("Your input:\t")
            embedded_input = create_embeddings_for_inference(
                user_input, embedding_model
            )
            fc_model_input = tf.convert_to_tensor(embedded_input, dtype=tf.float32)
            model_output = fc_model.predict(fc_model_input, verbose=0)
            if model_output[0, 1] * 100 > 50:
                print(f"Model output: \tSPAM!!!   {model_output[0, 1]:.2%}  ")
            else:
                print("Model output: \tNOT-SPAM!!!")

    except KeyboardInterrupt:
        print("\nThe inference has been stopped.\n")

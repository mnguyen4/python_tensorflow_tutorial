import collections
import pathlib
import re
import string

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

print("============================ Predict tag for Stack Overflow question ============================")
# Download the dataset
data_url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
dataset = utils.get_file("stack_overflow_16k.tar.gz", data_url, untar=True, cache_dir="stack_overflow", cache_subdir="")
dataset_dir = pathlib.Path(dataset).parent
print(list(dataset_dir.iterdir()))
train_dir = dataset_dir/'train'
print(list(train_dir.iterdir()))
sample_file = train_dir/'python/1755.txt'
with open(sample_file) as f:
    print(f.read())

# Load dataset from disk
batch_size = 32
seed = 42

raw_train_ds = preprocessing.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2, subset='training', seed=seed
)

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(10):
        print("Question: ", text_batch.numpy()[i][:100], '...')
        print("Label: ", label_batch.numpy()[i])

for i, label in enumerate(raw_train_ds.class_names):
    print("Label", i, "corresponds to", label)

# validation data set
raw_val_ds = preprocessing.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2, subset="validation", seed=seed
)

# test data set
test_dir = dataset_dir/'test'
raw_test_ds = preprocessing.text_dataset_from_directory(
    test_dir, batch_size=batch_size
)

print("============================ Prepare dataset for training ============================")
# standardization, tokenization, vectorization
VOCAB_SIZE = 10000
binary_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE, output_mode="binary"
)

MAX_SEQUENCE_LENGTH = 250
int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

train_text = raw_train_ds.map(lambda text, labels: text)
binary_vectorize_layer.adapt(train_text)
int_vectorize_layer.adapt(train_text)

def binary_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return binary_vectorize_layer(text), label

def int_vextorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return int_vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[0], label_batch[0]
print("Question", first_question)
print("Label", first_label)

print("'binary' vectorized question:", binary_vectorize_text(first_question, first_label)[0])
print("'int' vectorized question:", int_vextorize_text(first_question, first_label)[0])
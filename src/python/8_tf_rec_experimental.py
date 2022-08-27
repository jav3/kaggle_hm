from turtle import end_fill
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import pyarrow as pa


trans = pa.ipc.open_file('data/transactions_m.arrow').read_pandas()

# Select the basic features.
trans_2 = tf.data.Dataset.from_tensor_slices(dict(trans[['cid','aid']]))

ratings = trans_2.map(tf.autograph.experimental.do_not_convert(lambda x: {
    "aid": x["aid"],
    "user_id": x["cid"],
}))

aid = trans_2.map(tf.autograph.experimental.do_not_convert(lambda x: x["aid"]))

tf.random.set_seed(42)
shuffled = ratings.shuffle(100000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

aid_x = aid.batch(1_000)
user_ids = ratings.batch(1_000_000).map(tf.autograph.experimental.do_not_convert(lambda x: x["user_id"]))

unique_aid_x = trans.aid.unique()
unique_user_ids = trans.cid.unique()

# Model
embedding_dimension = 32

user_model = tf.keras.Sequential([
  tf.keras.layers.IntegerLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

aid_model = tf.keras.Sequential([
  tf.keras.layers.IntegerLookup(
      vocabulary=unique_aid_x, mask_token=None),
  tf.keras.layers.Embedding(len(unique_aid_x) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=aid.batch(128).map(aid_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class hmModel(tfrs.Model):

  def __init__(self, user_model, aid_model):
    super().__init__()
    self.aid_model: tf.keras.Model = aid_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the aid features and pass them into the aid model,
    # getting embeddings back.
    positive_aid_embeddings = self.aid_model(features["aid"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_aid_embeddings)

model = hmModel(user_model, aid_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=3)

model.evaluate(cached_test, return_dict=True)


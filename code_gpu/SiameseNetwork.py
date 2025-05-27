import matplotlib.pyplot as plt
import tensorflow as tf
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet
import data_generator
import utils
import numpy as np

target_shape = (120, 160)  
alpha = 5
beta = 1

# Using relative paths for datasets
train_dataset = data_generator.prepareTfDataset(
    "../dataset/Invalid",
    "../dataset/Valid", 18
)

test_dataset = data_generator.prepareTfDatasetTest(
    "../dataset/Invalid",
    "../dataset/Valid", 18
)

base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, pair1, pair2):
        pair_distance = ops.sum(tf.square(pair1 - pair2), -1)
        return pair_distance

pair1_input = layers.Input(name="first elem of the pair", shape=target_shape + (3,))
pair2_input = layers.Input(name="second elem of the pair", shape=target_shape + (3,))

distance = DistanceLayer()(
    embedding(resnet.preprocess_input(pair1_input)),
    embedding(resnet.preprocess_input(pair2_input)),
)

siamese_network = Model(inputs=[pair1_input, pair2_input], outputs=distance)

class SiameseModel(Model):
    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        x, y = data
        pair_distance = self.siamese_network(x)

        loss = (1 - y) * tf.square(pair_distance) + alpha * y * tf.square(
            tf.maximum(0.0, self.margin - pair_distance)
        )
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=10)

similarityl = []
labell = []
for i in test_dataset:
    pair, label = i
    p1_embedding, p2_embedding = (
        embedding(resnet.preprocess_input(pair[0])),
        embedding(resnet.preprocess_input(pair[1])),
    )

    cosine_similarity = metrics.CosineSimilarity()

    similarity = cosine_similarity(p1_embedding, p2_embedding)
    print("similarity:", similarity.numpy(), " and label is ", label)
    similarityl.append(similarity)
    labell.append(label.numpy())

utils.plotSimilarities(similarityl, labell)

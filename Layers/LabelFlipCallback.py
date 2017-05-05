import keras
from keras.models import Sequential
from keras.layers import Layer

from Layers import LabelFlipNoise


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.model = Sequential()
        for layer in self.model.layers:
            if isinstance(layer, LabelFlipNoise):
                layer = LabelFlipNoise()
                layer.trainable=True


    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
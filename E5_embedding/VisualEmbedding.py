from keras.engine import Input
from keras.layers import Dense
from keras.models import Sequential


class Embedder():

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim


    def model(self):
        # input_dim = 8000
        # output_dim = 400
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        model.add(Dense(self.output_dim, activation='relu')) # relu or sigmoid?
        return model

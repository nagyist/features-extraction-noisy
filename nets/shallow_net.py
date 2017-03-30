from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten
from keras.models import Sequential

from Layers import LabelFlipNoise
from Layers import OutlierNoise


class Hidden:
    def __init__(self, neurons, dropout=0):
        self.neurons = neurons
        self.dropout = dropout


def new_model(in_shape, out_shape, hiddens=[], lf=False, lf_decay=0.1, outl=False, outl_alpha=0.1, flatten=True):
    # type: (list, list, list(Hidden), bool, float) -> Sequential
    model = Sequential()

    input_done=False
    if flatten:
        model.add(Flatten(name="flatten", input_shape=in_shape))
        input_done=True
    for i, h in enumerate(hiddens):
        if input_done==False:
            model.add(Dense(h.neurons, activation='relu', name="additional_hidden_" + str(i), input_shape=in_shape))
        else:
            model.add(Dense(h.neurons, activation='relu', name="additional_hidden_" + str(i)))
            if h.dropout > 0:
                model.add(Dropout(h.dropout, name="additional_dropout_" + str(i)))
    if input_done == False:
        model.add(Dense(out_shape, name="dense", input_shape=in_shape))
    else:
        model.add(Dense(out_shape, name="dense"))

    model.add(Activation("softmax", name="softmax"))
    if lf:
        model.add(LabelFlipNoise(weight_decay=lf_decay, trainable=True))
    if outl:
        model.add(OutlierNoise(alpha=outl_alpha))
    return model


def save_model_json(model, out_file):
    # type: (keras.models.Model, str) -> None
    json = model.to_json()
    fjson = file(out_file , "w")
    fjson.write(json)
    fjson.close()
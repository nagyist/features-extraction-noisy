import os
from time import strftime, gmtime

import cPickle
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.models import Sequential

from Layers import LabelFlipNoise


class ShallowNet():



    def __init__(self):
        self.additional_neurons = None
        self.additional_dropout = None
        self.labelflip = None
        self.input_shape = None
        self.output_size = None
        self.lf_weight_decay = None
        self.lf_trainable = None
        self.model_name = None
        self.creation_date = None
        self.creation_time = None
        self.model_fname = None

    def init(self, input_shape, output_size, add_neur=0, add_drop=-1,
             labelflip=False, lf_wd=0.1, lf_train=False, name="shallow_model"):
        self.additional_neurons = add_neur
        self.additional_dropout = add_drop
        self.labelflip = labelflip
        self.input_shape = input_shape
        self.output_size = output_size
        self.lf_weight_decay = lf_wd
        self.lf_trainable = lf_train
        self.model_name = name
        self.model_fname = self.model_name + "_" + self.creation_date + "_" + self.creation_time
        return self


    def model(self):
        # type: () -> Sequential
        model = Sequential()
        model.add(Flatten(name="flatten", input_shape=self.input_shape))

        if self.additional_neurons > 0:
            model.add(Dense(self.additional_neurons, activation='relu', name="additional_hidden"))
            if self.additional_dropout > 0:
                model.add(Dropout(self.additional_dropout, name="additional_dropout"))
        model.add(Dense(self.output_size, name="dense"))  # for toy dataset
        model.add(Activation("softmax", name="softmax"))

        if self.labelflip:
            model.add(LabelFlipNoise(weight_decay=self.lf_weight_decay, trainable=self.lf_trainable))
        return model

    def get_model_pretrained(self, ext='.h5', dir_path=None):
        weight_file = self.get_weight_fname(ext, dir_path)
        model = self.model()
        model.load_weights(weight_file)
        return model

    def get_weight_fname(self, ext='.h5', dir_path=None):
        path = self.model_fname
        path += ext
        if dir_path is not None:
            path = os.path.join(dir_path, path)
        return path


    def save(self, dir_path=None):
        """save class as self.name.txt"""
        path = self.model_fname+'.cfg'
        if dir_path is not None:
            path = os.path.join(dir_path, path)
        file = open(path,'w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()



    def load(self, model_fname, dir_path=None):
        """try load self.name.txt"""
        path = model_fname+'.cfg'
        if dir_path is not None:
            path = os.path.join(dir_path, path)
        file = open(path,'r')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)


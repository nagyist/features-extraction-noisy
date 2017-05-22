import os

import jsonpickle
from keras.optimizers import Adam, Adadelta


class Config:
    # NB: YOU HAVE TO INSTALL SIMPLEJSON WITH: pip install simplejson
    __pickle_backend='simplejson'
    @staticmethod
    def loadJSON(json_str_or_path):
        jsonpickle.set_preferred_backend(Config.__pickle_backend)
        if isinstance(json_str_or_path, basestring):
            if os.path.exists(json_str_or_path):
                json_str_or_path = file(json_str_or_path, 'r').read()
            return jsonpickle.decode(json_str_or_path)
        else:
            return None



    def __init__(self,
                 lr=10,
                 bs=32,
                 epochs=50,
                 opt=Adadelta,
                 sp_dim=200,
                 tx_act='softmax',
                 im_act='tanh',
                 tx_hid=None,
                 im_hid=None,
                 contr_w=1,
                 contr_inv_w=1,
                 log_w_tx=1,
                 log_w_im=0,
                 w_init='glorot_uniform'):
        self.lr = lr
        self.bs = bs
        self.epochs = epochs
        self.opt = opt
        self.sp_dim= sp_dim
        self.tx_act = tx_act
        self.im_act = im_act
        self.tx_hid = tx_hid
        self.im_hid = im_hid
        self.contr_w = contr_w
        self.contr_inv_w = contr_inv_w
        self.log_w_tx = log_w_tx
        self.log_w_im = log_w_im
        self.w_init=w_init

    def toJSON(self, human_readable=True):
        jsonpickle.set_preferred_backend(Config.__pickle_backend)
        if human_readable:
            jsonpickle.set_encoder_options(Config.__pickle_backend, sort_keys=True, indent=4)
        return jsonpickle.encode(self)


    def saveJSON(self, file_path):
        f = file(file_path, 'w')
        f.write(self.toJSON())
        f.close()

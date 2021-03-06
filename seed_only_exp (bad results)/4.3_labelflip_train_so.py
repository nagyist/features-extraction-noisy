import sys

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD

import config
from Layers import LabelFlipNoise
from config import common

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32



def main(args):
    config.init()



    feat_net = 'resnet50'
    print("")
    print("")
    print("Running experiment on net: " + feat_net)


    trainset_name = config.DATASET + '_so'
    trainset = common.feat_dataset(trainset_name, feat_net)
    validset = common.feat_dataset(config.DATASET + '_so_test', feat_net)
    valid_data = validset.data, validset.getLabelsVec()
    valid_split = 0

    in_shape = config.feat_shape_dict[feat_net]
    out_shape = trainset.labelsize


    def addestra(model, name, optimizer, epochs, callbacks, chk_period=-1, loss_in_name=False):
        shallow_path = common.shallow_path(name, trainset_name, feat_net, ext=False)

        if chk_period > 0:
            name = shallow_path + '.weights.{epoch:02d}' + ('-{val_loss:.2f}.h5' if loss_in_name else '.h5')
            checkpoint = ModelCheckpoint(name, monitor='val_acc', save_weights_only=True, period=chk_period)
            callbacks.append(checkpoint)

        bestpoint = ModelCheckpoint(shallow_path + '.weights.best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
        callbacks.append(bestpoint)

        model.compile(optimizer=optimizer, loss=LOSS, metrics=METRIC)
        #model.summary()
        #print("Valid split: " + str(valid_split))
        model.fit(trainset.data, trainset.getLabelsVec(), nb_epoch=epochs, batch_size=BATCH, callbacks=callbacks,
                  shuffle=True, validation_data=valid_data, validation_split=valid_split)


        save_model_json(model, shallow_path + '.json')
        model.save_weights(shallow_path + '.weights.last.h5')


    def for_resnet50():

        early_stopping = EarlyStopping('val_loss', min_delta=0.01, patience=7, verbose=1)
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, epsilon=0.01, cooldown=0, min_lr=0)
        callbacks = [early_stopping, reduceLR]

        A = new_model(in_shape, out_shape)
        optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        addestra(A, "A_5ep", optimizer , 100, callbacks, chk_period=1, loss_in_name=True)




        shallow_path = common.shallow_path("A_5ep", trainset_name, feat_net, ext=False)
        early_stopping = EarlyStopping('val_loss', min_delta=0.001, patience=10, verbose=1)
        reduceLR = ReduceLROnPlateau('val_loss', factor=0.1, patience=4, verbose=1, epsilon=0.0001)
        callbacks = [early_stopping, reduceLR]

        LF = new_model(in_shape, out_shape, lf=True, lf_decay=0.03)
        LF.load_weights(shallow_path + '.weights.best.h5',by_name=True)
        optimizer = SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True)
        addestra(LF, "LF_FT_A", optimizer, epochs=100, callbacks=callbacks, chk_period=1)



    def for_vgg():
        pass
        # m = new_model(in_shape, out_shape, hiddens=[Hidden(4096, 0.5), Hidden(4096, 0.5)])
        # addestra(m, "H4K_H4K", SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)
        #
        # m = new_model(in_shape, out_shape, hiddens=[Hidden(4096, 0.5)])
        # addestra(m, "H4K", SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)


    if feat_net == 'resnet50':
        for_resnet50()
    if feat_net.startswith('vgg'):
        for_vgg()







class Hidden:
    def __init__(self, neurons, dropout=0):
        self.neurons = neurons
        self.dropout = dropout


def new_model(in_shape, out_shape, hiddens=[], lf=False, lf_decay=0.1):
    # type: (list, list, list(Hidden), bool, float) -> Sequential
    model = Sequential()
    model.add(Flatten(name="flatten", input_shape=in_shape))

    for i, h in enumerate(hiddens):
        model.add(Dense(h.neurons, activation='relu', name="additional_hidden_" + str(i)))
        if h.dropout > 0:
            model.add(Dropout(h.dropout, name="additional_dropout_" + str(i)))
    model.add(Dense(out_shape, name="dense"))
    model.add(Activation("softmax", name="softmax"))
    if lf:
        model.add(LabelFlipNoise(weight_decay=lf_decay, trainable=True))
    return model


def save_model_json(model, out_file):
    # type: (keras.models.Model, str) -> None
    json = model.to_json()
    fjson = file(out_file , "w")
    fjson.write(json)
    fjson.close()



if __name__ == "__main__":
    main(sys.argv)






#
# parser = ArgumentParser()
# parser.add_argument("-n", "-net", action='store', dest='net_list', required=True, type=str, nargs='+',
#                     choices=["vgg16", "vgg19", "resnet50", "inception_v3"],
#                     help="Set the net(s) to use for the training experiment.")
# # parser.add_argument("-train", action='store_true', dest='train', default=False,
# #                     help="Train experiment.")
# # parser.add_argument("-test", action='store_true', dest='test', default=False,
# #                     help="Test experiment.")
# # parser.add_argument("-class-test", action='store_true', dest='class_test', default=False,
# #                     help="Test experiment.")
# parser.add_argument("-l", "--local", action='store_true', dest='use_local', default=False,
#                     help="Test experiment.")
# parser.add_argument("-s", "--scratch", action='store_false', dest='use_local', default=False,
#                     help="Test experiment.")
# args = parser.parse_args()
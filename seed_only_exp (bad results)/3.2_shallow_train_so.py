
import sys
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import cfg
import common
from Layers import LabelFlipNoise
from imdataset import ImageDataset
from nets import Hidden
from nets import new_model
from nets import save_model_json

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32
SHUFFLE = True if cfg.USE_TOY_DATASET else False



def main(args):
    cfg.init()









    feat_net = 'resnet50'
    print("")
    print("")
    print("Running experiment on net: " + feat_net)




    trainset_name = cfg.DATASET + '_so'
    trainset = common.feat_dataset(trainset_name, feat_net)
    validset = common.feat_dataset( cfg.DATASET + '_so_test', feat_net)
    valid_data = validset.data, validset.getLabelsVec()
    valid_split = 0

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = trainset.labelsize



    def addestra(model, name, optimizer, epochs, callbacks, chk_period=-1, loss_in_name=False):
        shallow_path = common.shallow_path(name, trainset_name, feat_net, ext=False)

        if chk_period > 0:
            name = shallow_path + '.weights.{epoch:02d}' + ('-{val_loss:.2f}.h5' if loss_in_name else '.h5')
            checkpoint = ModelCheckpoint(name, monitor='val_loss', save_weights_only=True, period=chk_period)
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
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=7, verbose=1)
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                                     epsilon=0.0001, cooldown=0, min_lr=0)
        callbacks = [early_stopping, reduceLR]

        m = new_model(in_shape, out_shape)
        addestra(m, "A", SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)

        m = new_model(in_shape, out_shape, hiddens=[Hidden(4096, 0.5)])
        addestra(m, "H4K", SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)

        m = new_model(in_shape, out_shape, hiddens=[Hidden(4096, 0.5), Hidden(4096, 0.5)])
        addestra(m, "H4K_H4K", SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)



    def for_vgg():
        if feat_net.endswith('fc1'):
            flatten=False
        else:
            flatten=True
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=15, verbose=1)
        reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1,
                                                     epsilon=0.001, cooldown=0, min_lr=0)
        callbacks = [early_stopping, reduceLR]

        m = new_model(in_shape, out_shape, hiddens=[Hidden(4096, 0.5)],flatten=flatten)
        addestra(m, "A_so", SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)

        m = new_model(in_shape, out_shape, hiddens=[Hidden(4096, 0.5)],flatten=flatten)
        addestra(m, "H4K_so", SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)

        m = new_model(in_shape, out_shape, hiddens=[Hidden(4096, 0.5), Hidden(4096, 0.5)],flatten=flatten)
        addestra(m, "H4K_H4K_so", SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True), epochs=100, callbacks=callbacks)



    if feat_net == 'resnet50':
        for_resnet50()
    if feat_net.startswith('vgg'):
        for_vgg()






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
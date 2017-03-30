
import sys
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.optimizers import SGD

from E3_shallow.Shallow import ShallowTrainer, ShallowNet, ShallowNetBuilder
from cfg import cfg
import common

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32


def main(args):
    cfg.init()
    shallow_train('resnet50')



def shallow_train(feat_net='resnet50', double_seeds=True, outlier_model=False):

    if cfg.USE_TOY_DATASET:
        trainset_name = cfg.DATASET + '_train'
        trainset = common.feat_dataset(trainset_name, feat_net)
        trainset.shuffle()
        valid_split = 0.3
    else:
        trainset_name = cfg.DATASET + '_train' + ('_ds' if double_seeds else '')
        trainset = common.feat_dataset(trainset_name, feat_net)
        validset = common.feat_dataset( cfg.DATASET + '_valid', feat_net)
        valid_split = 0

    print("Shallow train on features from net: " + feat_net)
    print("Trainset: " + trainset_name)

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = trainset.labelsize

    ST = ShallowTrainer(feat_net, trainset_name, validset, valid_split, batch_size=BATCH, loss=LOSS, metric=METRIC)
    SNB = ShallowNetBuilder(in_shape, out_shape)



    if feat_net == 'resnet50':
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=15, verbose=1)
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=0.001)
        callbacks = [early_stopping, reduceLR]

        opt = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        ST.train(SNB.H8K(), opt, epochs=100, callbacks=callbacks, chk_period=1)
        ST.train(SNB.H4K(), opt, epochs=100, callbacks=callbacks, chk_period=1)
        ST.train(SNB.A(), opt, epochs=100, callbacks=callbacks, chk_period=1)


    if feat_net.startswith('vgg'):
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=15, verbose=1)
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=0.001)
        callbacks = [early_stopping, reduceLR]

        opt = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
        ST.train(SNB.H8K(), opt, epochs=100, callbacks=callbacks, chk_period=1)
        ST.train(SNB.H4K(), opt, epochs=100, callbacks=callbacks, chk_period=1)
        ST.train(SNB.A(), opt, epochs=100, callbacks=callbacks, chk_period=1)



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
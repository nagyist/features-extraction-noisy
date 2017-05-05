import sys

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD

from E3_shallow.Shallow import ShallowTrainer, ShallowNetBuilder, ShallowLoader
from config import cfg, common, feat_dataset_n_classes

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32


def main(args):
    cfg.init()
    shallow_finetune_labelflip('resnet50')



def shallow_finetune_labelflip(feat_net='resnet50', double_seeds=True, outlier_model=False):

    if cfg.use_toy_dataset:
        trainset_name = cfg.dataset + '_train'
        validset_name = None
        valid_split = 0.3
    else:
        trainset_name = cfg.dataset + '_train' + ('_ds' if double_seeds else '')
        validset_name = cfg.dataset + '_valid'
        valid_split = 0

    print("Shallow train on features from net: " + feat_net)
    print("Trainset: " + trainset_name)

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = feat_dataset_n_classes(trainset_name, feat_net)




    SNB = ShallowNetBuilder(in_shape, out_shape)
    SL = ShallowLoader(trainset_name, feat_net)

    load_iter = ['10']
    decay = [0.1, 0.01, 0.001, 0.0001]
    lrs = [0.01, 0.001, 0.0001, 0.00001]

    for li in load_iter:
        for dc in decay:
            for lr in lrs:
                extr_n = '_ft@{}_dc-{}_lr-{}'.format(li, dc, lr)
                opt = SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True)
                snet = [SNB.H8K(extr_n, lf_decay=0.01).init(lf=True).load(SL, li, SNB.H8K())]
                ST = ShallowTrainer(feat_net, trainset_name, validset_name, valid_split, batch_size=BATCH, loss=LOSS, metric=METRIC)
                ST.train(snet, opt, epochs=20, chk_period=1)





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

import sys
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import cfg
import common
from Layers import LabelFlipNoise
from imdataset import ImageDataset
from test import test_net, write_net_score

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32



def main(args):
    cfg.init()



    feat_net = 'resnet50'
    print("")
    print("")
    print("Running experiment on net: " + feat_net)


    # if cfg.USE_TOY_DATASET:
    #     trainset_name = cfg.DATASET + '_train'
    # else:
    #     trainset_name = cfg.DATASET + '_train_ds'

    testset_name = 'dbp3120_test_so'
    testset = common.feat_dataset(testset_name, feat_net)

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = testset.labelsize



    def for_resnet50():


        #shallow_path = common.shallow_path("LF_FT_A", trainset_name, feat_net, ext=False)
        LF = new_model(in_shape, out_shape)

        shallow_path = cfg.SHALLOW_PATH + "shallow_A__feat_dbp3120_so__resnet50__avg_pool.weights.best.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "A best", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_A__feat_dbp3120_so__resnet50__avg_pool.weights.last.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "A last", testset_name, "test_results.csv", detailed_csv=True)


        shallow_path = cfg.SHALLOW_PATH + "shallow_A_5ep__feat_dbp3120_so__resnet50__avg_pool.weights.best.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "A_5ep best", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_A_5ep__feat_dbp3120_so__resnet50__avg_pool.weights.last.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "A_5ep last", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_A__feat_dbp3120_so__resnet50__avg_pool.weights.best.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "LF A best", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_A__feat_dbp3120_so__resnet50__avg_pool.weights.00.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "LF A 0", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_A__feat_dbp3120_so__resnet50__avg_pool.weights.01.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "LF A 1", testset_name, "test_results.csv", detailed_csv=True)


        shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_A__feat_dbp3120_so__resnet50__avg_pool.weights.02.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "LF A 2", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_A__feat_dbp3120_so__resnet50__avg_pool.weights.03.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "LF A 3", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_A__feat_dbp3120_so__resnet50__avg_pool.weights.04.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "LF A 4", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_A__feat_dbp3120_so__resnet50__avg_pool.weights.05.h5"
        LF.load_weights(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "LF A 5", testset_name, "test_results.csv", detailed_csv=True)

    if feat_net == 'resnet50':
        for_resnet50()


#
# GLOBAL SCORES
# top1 score: 0.256198347107
# top5 score: 0.376033057851
# precision:  0.734159779614
# recall:     0.256198347107
# f1 score :  0.318400588079
# examples:   242


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
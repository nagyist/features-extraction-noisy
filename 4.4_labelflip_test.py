
import sys
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import cfg
import common
from Layers import LabelFlipNoise
from imdataset import ImageDataset
from nets import new_model
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

    testset_name = cfg.DATASET + '_test' + '_outl'


    testset = common.feat_dataset(testset_name, feat_net)

    in_shape = cfg.feat_shape_dict[feat_net]
    out_shape = testset.labelsize




    def for_resnet50_AB():
        # shallow_path = common.shallow_path("LF_FT_A", trainset_name, feat_net, ext=False)
        LF = new_model(in_shape, out_shape)


        shallow_path = cfg.SHALLOW_PATH + "shallow_AB__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.best.h5"
        LF.load(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "AB best", testset_name, "test_results.csv", detailed_csv=True)

        # shallow_path = cfg.SHALLOW_PATH + "shallow_AB__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.last.h5"
        # LF.load_weights(shallow_path, by_name=True)
        # score = test_net(LF, testset)
        # write_net_score(score, "AB last", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_AB__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.05.h5"
        LF.load(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "AB 5", testset_name, "test_results.csv", detailed_csv=True)


        shallow_path = cfg.SHALLOW_PATH + "shallow_AB__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.10.h5"
        LF.load(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "AB 10", testset_name, "test_results.csv", detailed_csv=True)



        lf_index = ['00', '01', '02', '03', '04', '05', '07', '09', '11', '13' , '15', '17', '20', '22', '24', '26', '30', '35', '40']

        for lfi in lf_index:
            shallow_path = cfg.SHALLOW_PATH + "shallow_OL_FT_AB__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.{}.h5".format(lfi)
            LF.load(shallow_path, by_name=True)
            score = test_net(LF, testset)
            write_net_score(score, "OL AB " + lfi, testset_name, "test_results.csv", detailed_csv=True)




    def for_resnet50_HL():
        # shallow_path = common.shallow_path("LF_FT_A", trainset_name, feat_net, ext=False)
        LF = new_model(in_shape, out_shape, hiddens=[Hidden(8000, 0.5)])


        shallow_path = cfg.SHALLOW_PATH + "shallow_H8K__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.best.h5"
        LF.load(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "H8K best", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_H8K__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.last.h5"
        LF.load(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "H8K last", testset_name, "test_results.csv", detailed_csv=True)

        shallow_path = cfg.SHALLOW_PATH + "shallow_H8K__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.05.h5"
        LF.load(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "H8K 5", testset_name, "test_results.csv", detailed_csv=True)


        shallow_path = cfg.SHALLOW_PATH + "shallow_H8K__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.10.h5"
        LF.load(shallow_path, by_name=True)
        score = test_net(LF, testset)
        write_net_score(score, "H8K 10", testset_name, "test_results.csv", detailed_csv=True)



        lf_index = ['00', '01', '02', '03', '04', '05', '07', '09', '11', '13' , '15', '17', '20']

        for lfi in lf_index:
            shallow_path = cfg.SHALLOW_PATH + "shallow_LF_FT_H8K__feat_dbp3120_train_ds_outl__resnet50__avg_pool.weights.{}.h5".format(lfi)
            LF.load(shallow_path, by_name=True)
            score = test_net(LF, testset)
            write_net_score(score, "LF H8K " + lfi, testset_name, "test_results.csv", detailed_csv=True)



    if feat_net == 'resnet50':
        for_resnet50_AB()
        #for_resnet50_HL()

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
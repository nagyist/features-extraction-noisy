import codecs
import copy
import os
import random
import sys

from theano.gradient import np

from E5_embedding import cfg_emb
from E5_embedding.cfg_emb import VISUAL_FEATURES, CLASS_LIST, get_class_list, TEXT_FEATURES_TRAIN_400
from config import cfg

from imdataset import ImageDataset
from imdataset.ImageDataset import SplitOptions

LOSS = 'categorical_crossentropy'
METRIC = ['accuracy']
BATCH = 32


def split_train_valid_test(visual_features=VISUAL_FEATURES, text_features=TEXT_FEATURES_TRAIN_400, n_test_classes=100):


    visual_features = ImageDataset().load_hdf5(visual_features)




    class_list = cfg_emb.load_class_list(cfg_emb.CLASS_LIST)
    class_name_list = cfg_emb.load_class_list(cfg_emb.CLASS_NAME_LIST)

    class_list = np.asarray(class_list, dtype=np.int32)
    class_name_list = np.asarray(class_name_list)

    class_permutation = np.random.permutation(range(0, len(class_list)))

    class_list_test = class_list[class_permutation][0:n_test_classes]
    class_list_train = class_list[class_permutation][n_test_classes:-1]
    class_name_list_test = class_name_list[class_permutation][0:n_test_classes]
    class_name_list_train = class_name_list[class_permutation][n_test_classes:-1]

    test_sort = np.argsort(class_list_test)
    train_sort = np.argsort(class_list_train)

    class_list_test = class_list_test[test_sort]
    class_list_train = class_list_train[train_sort]
    class_name_list_test = class_name_list_test[test_sort]
    class_name_list_train = class_name_list_train[train_sort]



    print("Loading textual features..")
    text_features = np.load(text_features)

    text_features_test = text_features[class_permutation][0:n_test_classes][test_sort]
    text_features_train = text_features[class_permutation][n_test_classes:-1][train_sort]


    visual_features_test = visual_features.sub_dataset_with_labels(class_list_test)
    visual_features_train_valid = visual_features.sub_dataset_with_labels(class_list_train)

    split_options = [SplitOptions("flickr", 0.25), SplitOptions("google", 0.3)]
    exclude_file_starting_with = ["seed"]
    visual_features_train, visual_features_valid = \
        visual_features_train_valid.validation_per_class_split(split_options, exclude_file_starting_with)



    cfg_emb.save_class_list(class_list_test, cfg_emb.CLASS_LIST_TEST)
    cfg_emb.save_class_list(class_list_train, cfg_emb.CLASS_LIST_TRAIN)

    cfg_emb.save_class_list(class_name_list_test, cfg_emb.CLASS_NAME_LIST_TEST)
    cfg_emb.save_class_list(class_name_list_train, cfg_emb.CLASS_NAME_LIST_TRAIN)

    np.save(cfg_emb.TEXT_FEATURES_TEST, text_features_test)
    np.save(cfg_emb.TEXT_FEATURES_TRAIN, text_features_train)

    visual_features_train.save_hdf5(cfg_emb.VISUAL_FEATURES_TRAIN)
    visual_features_valid.save_hdf5(cfg_emb.VISUAL_FEATURES_VALID)
    visual_features_test.save_hdf5(cfg_emb.VISUAL_FEATURES_TEST)


def main(args):
    cfg.init()
    split_train_valid_test()





if __name__ == "__main__":
    main(sys.argv)



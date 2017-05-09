import codecs
import copy
import os

from E3_shallow.Shallow import ShallowNetBuilder, ShallowLoader, ShallowTester
from config import cfg, common, feat_dataset_n_classes
from imdataset import ImageDataset





cfg.init()
DOUBLE_SEEDS = True
FEAT_NET = 'resnet50'
FEAT_DATASET = cfg.dataset
FEAT_TESTSET_NAME = FEAT_DATASET
FEAT_TRAINSET_NAME = cfg.dataset + '_train' + ('_ds' if DOUBLE_SEEDS else '')

PRUNING_KEEP_N_CLASSES = 500
VISUAL_FEATURES = "extracted_features/feat_dbp3120__resnet50__avg_pool_pruned-A@best_nb-classes-500_test-on-dbp3120.h5"
# "shallow_extracted_features/shallow_feat_dbp3120_train_ds.h5"

IM2DOC_PREDICTION_FOLDER = 'im2doc_prediction'
IM2DOC_MODEL_FOLDER = "im2doc_embedding"

if not os.path.isdir(IM2DOC_MODEL_FOLDER):
    os.mkdir(IM2DOC_MODEL_FOLDER)

if not os.path.isdir(IM2DOC_PREDICTION_FOLDER):
    os.mkdir(IM2DOC_PREDICTION_FOLDER)


CLASS_LIST = "class_keep_from_pruning.txt"
CLASS_LIST_TEST = "class_keep_from_pruning-test.txt"
CLASS_LIST_TRAIN = "class_keep_from_pruning-train.txt"

CLASS_NAME_LIST = "class_names_keep_from_pruning.txt"
CLASS_NAME_LIST_TEST = "class_names_keep_from_pruning-test.txt"
CLASS_NAME_LIST_TRAIN = "class_names_keep_from_pruning-train.txt"

# TEXT_FEATURES_TRAIN_400 = "docvec_400_train_on_400.npy"
# TEXT_FEATURES_TOTAL_500 = "docvec_500_train_on_500.npy"
TEXT_FEATURES_TRAIN_400 = "docvec_400_train_on_wiki.npy"
TEXT_FEATURES_TOTAL_500 = "docvec_500_train_on_wiki.npy"
_TEXT_FEATURES_TEST_100 = None

def GET_TEXT_FEATURES_TEST_100():
    import numpy as np
    import cfg_emb
    if cfg_emb._TEXT_FEATURES_TEST_100 is None:
        docs_vectors_500 = np.load(TEXT_FEATURES_TOTAL_500)
        docs_vectors_100_zero_shot = []
        class_list_all = load_class_list(CLASS_LIST)
        class_list_for_map = load_class_list(CLASS_LIST_TEST)
        for i, cls in enumerate(class_list_all):
            if cls in class_list_for_map:
                docs_vectors_100_zero_shot.append(docs_vectors_500[i])
        cfg_emb._TEXT_FEATURES_TEST_100 = np.asarray(docs_vectors_100_zero_shot)
    return cfg_emb._TEXT_FEATURES_TEST_100


# TEXT_FEATURES_TEST = "doc2vec_dbpedia_vectors-test.npy"
# TEXT_FEATURES_TRAIN = "doc2vec_dbpedia_vectors-train.npy"





SHALLOW_WEIGHT_LOAD = 'best'
USE_LABELFLIP = False
SHALLOW_FT_LF_WEIGHT_LOAD = '00'
LF_DECAY=0.01



in_shape = cfg.feat_shape_dict[FEAT_NET]
out_shape = feat_dataset_n_classes(FEAT_TESTSET_NAME, FEAT_NET)

SNB = ShallowNetBuilder(in_shape, out_shape)
SL = ShallowLoader(FEAT_TRAINSET_NAME, FEAT_NET)
#ST = ShallowTester(FEAT_NET, FEAT_TRAINSET_NAME, FEAT_TESTSET_NAME, csv_class_stats=False, csv_global_stats=False)

# Nets to test
# shallow_nets = [SNB.H8K]
SHALLOW_NET_BUILD = SNB.A
if USE_LABELFLIP:
    raise ValueError("Not implemented...")
else:
    SHALLOW_NET = SHALLOW_NET_BUILD().init().load(SL, SHALLOW_WEIGHT_LOAD)



def pruned_feat_dataset(dataset_name, feat_net, shallow_net):
    return ImageDataset().load_hdf5( pruned_feat_dataset_path(dataset_name, feat_net, shallow_net))

def pruned_feat_dataset_path(dataset_name, testing_dataset_name, nb_classes, feat_net, shallow_net, ext=True):
    feat_path = common.feat_path(dataset_name, feat_net, ext=False)
    ret =  feat_path + '_pruned-'+shallow_net.name+'@'+shallow_net.weights_loaded_index+ '_nb-classes-' + str(nb_classes) + '_test-on-' + testing_dataset_name
    return ret + '.h5' if ext else ret



VISUAL_FEATURES = pruned_feat_dataset_path(FEAT_DATASET, FEAT_TESTSET_NAME, PRUNING_KEEP_N_CLASSES, FEAT_NET, SHALLOW_NET, False)
VISUAL_FEATURES_TRAIN = VISUAL_FEATURES + '__train.h5'
VISUAL_FEATURES_VALID = VISUAL_FEATURES + '__valid.h5'
VISUAL_FEATURES_TEST = VISUAL_FEATURES + '__test.h5'
VISUAL_FEATURES += '.h5'



__class_list = None

def get_class_list():
    import cfg_emb
    if cfg_emb.__class_list is None:
        cfg_emb.__class_list = file(CLASS_LIST, 'r').read().split('\n')
        # cfg_emb.__class_list.sort()
    return copy.deepcopy(cfg_emb.__class_list)


def load_class_list(path,  encoding='utf-8'):
    if path is not None:
        return codecs.open(path, 'r', encoding=encoding).read().split('\n')
    else:
        return None

def save_class_list(class_list, path, encoding='utf-8'):
    # type: (list, basestring) -> None
    f = codecs.open(path, 'w', encoding=encoding)
    for cls in class_list:
        f.write(unicode(cls) + '\n')
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.close()
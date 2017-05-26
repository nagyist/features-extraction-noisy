import os
# floatX=float32,device=gpu1,lib.cnmem=1,
# os.environ['THEANO_FLAGS'] = "exception_verbosity=high, optimizer=None"
from keras.engine import Model
from keras.models import load_model

from zzE6_joint_model.test_joint import retrieve_image_map, retrive_text_map, recall_top_k

os.environ['KERAS_BACKEND'] = "tensorflow"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# FORCE TENSORFLOW BACKEND HERE!! WITH THEANO THIS EXPERIMENT WON'T RUN!!
# NB: The problem of low performance of tensorflow backend with theano ordering is true only for convolutional layers!
#     We don't have to worry about that here.



from keras.callbacks import TensorBoard, ModelCheckpoint
from theano.gof import optimizer



import sys

from keras.optimizers import Adadelta

from E5_embedding import cfg_emb
from mAP_callback import ModelMAP
from JointEmbedding import JointEmbedder
from imdataset import ImageDataset
import numpy as np

def main():
    joint_embedding_train()

def joint_embedding_train(visual_features=cfg_emb.VISUAL_FEATURES_TRAIN,
                          text_features=cfg_emb.TEXT_FEATURES_400,
                          class_list=cfg_emb.CLASS_LIST_400,
                          visual_features_valid=cfg_emb.VISUAL_FEATURES_VALID,
                          visual_features_zs_test=cfg_emb.VISUAL_FEATURES_TEST,
                          text_features_zs_test=cfg_emb.TEXT_FEATURES_100,
                          class_list_test=cfg_emb.CLASS_LIST_100):
    import numpy as np

    print("Loading visual features..")
    visual_features = ImageDataset().load_hdf5(visual_features)
    if visual_features_valid is not None:
        visual_features_valid = ImageDataset().load_hdf5(visual_features_valid)

    print("Loading textual features..")
    if not isinstance(text_features, np.ndarray):
        text_features = np.load(text_features)
    if not isinstance(text_features_zs_test, np.ndarray) and text_features_zs_test is not None:
        text_features_zs_test = np.load(text_features_zs_test)

    if class_list is None:
        class_list = np.unique(visual_features.labels).tolist()
    else:
        class_list = cfg_emb.load_class_list(class_list, int_cast=True)

    if not isinstance(class_list_test, list):
        class_list_test = cfg_emb.load_class_list(class_list_test, int_cast=True)


    print("Generating dataset..")

    if class_list is not None:
        cycle_clslst_txfeat = class_list, text_features
    else:
        cycle_clslst_txfeat = enumerate(text_features)


    im_data_train = []
    tx_data_train = []
    label_train = []
    if visual_features_valid is not None:
        im_data_valid = []
        tx_data_valid = []
        label_valid = []

    for lbl, docv in zip(cycle_clslst_txfeat[0], cycle_clslst_txfeat[1]):
        lbl = int(lbl)
        norm_docv = docv/np.linalg.norm(docv)  # l2 normalization

        visual_features_with_label = visual_features.sub_dataset_with_label(lbl)
        for visual_feat in visual_features_with_label.data:
            im_data_train.append(visual_feat)
            tx_data_train.append(norm_docv)
            label_train.append(lbl)

        if visual_features_valid is not None:
            visual_features_valid_with_label = visual_features_valid.sub_dataset_with_label(lbl)
            for visual_feat in visual_features_valid_with_label.data:
                im_data_valid.append(visual_feat)
                tx_data_valid.append(norm_docv)
                label_valid.append(lbl)

    # Image data conversion
    im_data_train = np.asarray(im_data_train)
    im_data_valid = np.asarray(im_data_valid)
    while len(im_data_train.shape) > 2:
        if im_data_train.shape[-1] == 1:
            im_data_train = np.squeeze(im_data_train, axis=(-1,))
    while len(im_data_valid.shape) > 2:
        if im_data_valid.shape[-1] == 1:
            im_data_valid = np.squeeze(im_data_valid, axis=(-1,))

    # Text data conversion
    tx_data_train = np.asarray(tx_data_train)
    tx_data_valid = np.asarray(tx_data_valid)
    while len(tx_data_train.shape) > 2:
        if tx_data_train.shape[-1] == 1:
            tx_data_train = np.squeeze(tx_data_train, axis=(-1,))
    while len(tx_data_valid.shape) > 2:
        if tx_data_valid.shape[-1] == 1:
            tx_data_valid = np.squeeze(tx_data_valid, axis=(-1,))

    # Label conversion
    label_train = np.asarray(label_train)
    label_valid = np.asarray(label_valid)

    while len(label_train.shape) > 2:
        if label_train.shape[-1] == 1:
            label_train = np.squeeze(label_train, axis=(-1,))
    while len(label_valid.shape) > 2:
        if label_valid.shape[-1] == 1:
            label_valid = np.squeeze(label_valid, axis=(-1,))

    print("Loading model..")

    path = 'im2doc_embedding/jointmodel_opt-adadelta_lr-100_bs-64-clamoroso?/jointmodel_opt-adadelta_lr-100_bs-64'
    path = 'im2doc_embedding/jointmodel_opt-adadelta_lr-100_bs-64/jointmodel_opt-adadelta_lr-100_bs-64'


    model_path = path + '.model.best.h5'
    weight_path = path + '.weights.49.h5'

    model = JointEmbedder.load_model(model_path=model_path, weight_path=weight_path)
    top_k = 10

    print("\nTest traning: ")
    map = retrieve_text_map(visual_features, text_features, class_list, joint_model=model)
    recall = recall_top_k(visual_features, text_features, class_list, joint_model=model,top_k=top_k)
    print("mAP = " + str(map))
    print("recall@{} = {}".format(top_k, recall))

    print("\nTest validation: ")
    map = retrieve_text_map(visual_features_valid, text_features, class_list, joint_model=model)
    recall = recall_top_k(visual_features_valid, text_features, class_list, joint_model=model,top_k=top_k)
    print("mAP = " + str(map))
    print("recall@{} = {}".format(top_k, recall))

    print("\nTest zero shot test: ")
    #map = test_joint_map(visual_features_zs_test, text_features_zs_test, class_list_test, joint_model=model)
    map = retrieve_text_map(visual_features_zs_test, text_features_zs_test, class_list_test, joint_model=model)
    recall = recall_top_k(visual_features_zs_test, text_features_zs_test, class_list_test, joint_model=model,
                       top_k=top_k)
    print("mAP = " + str(map))
    print("recall@{} = {}".format(top_k, recall))

    #
    # print("\n\n***********************************************\n\n")
    # for i in range(0,50):
    #     model.load_weights(path+'.weights.{}.h5'.format(str(i).zfill(2)))
    #     print("\nTest zero shot test (weights {}): ".format(str(i).zfill(2)))
    #     map = test_joint_map(visual_features_zs_test, text_features_zs_test, class_list_test, joint_model=model)
    #     print("mAP = " + str(map))
if __name__ == "__main__":
    main()

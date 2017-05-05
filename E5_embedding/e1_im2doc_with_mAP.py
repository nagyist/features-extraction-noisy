import json
import os
import sys
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.engine import Input
from keras.layers import Dense, copy
from keras.metrics import cosine_proximity
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop

from E5_embedding import cfg_emb
from config import cfg, common
from imdataset import ImageDataset
from mAP_callback import ModelMAP

# "shallow_extracted_features/shallow_feat_dbp3120_train_ds.h5"

MONITOR = 'loss'
#MONITOR = 'val_loss'


def main(args):
    im2docvec_wvalid_map()

def im2docvec_wvalid_map(visual_features=cfg_emb.VISUAL_FEATURES_TRAIN,
                         text_features=cfg_emb.TEXT_FEATURES_TRAIN_400,
                         class_list=cfg_emb.CLASS_LIST_TRAIN,

                         visual_features_valid=cfg_emb.VISUAL_FEATURES_VALID,

                         visual_features_zs_test=cfg_emb.VISUAL_FEATURES_TEST,
                         text_features_zs_test=cfg_emb.GET_TEXT_FEATURES_TEST_100(),
                         class_list_test=cfg_emb.CLASS_LIST_TEST):
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
        class_list = cfg_emb.load_class_list(class_list)

    if not isinstance(class_list_test, list):
        class_list_test = cfg_emb.load_class_list(class_list_test)


    print("Generating dataset..")

    if class_list is not None:
        cycle_on = class_list, text_features
    else:
        cycle_on = enumerate(text_features)


    data_train = []
    target_train = []
    if visual_features_valid is not None:
        data_valid = []
        target_valid = []

    for lbl, docv in zip(cycle_on[0], cycle_on[1]):
        lbl = int(lbl)
        norm_docv = docv/np.linalg.norm(docv)  # l2 normalization
        visual_features_with_label = visual_features.sub_dataset_with_label(lbl)
        for visual_feat in visual_features_with_label.data:
            data_train.append(visual_feat)
            target_train.append(norm_docv)

        if visual_features_valid is not None:
            visual_features_valid_with_label = visual_features_valid.sub_dataset_with_label(lbl)
            for visual_feat in visual_features_valid_with_label.data:
                data_valid.append(visual_feat)
                target_valid.append(norm_docv)

    data_train = np.asarray(data_train)
    data_valid = np.asarray(data_valid)

    while len(data_train.shape) > 2:
        if data_train.shape[-1] == 1:
            data_train = np.squeeze(data_train, axis=(-1,))

    while len(data_valid.shape) > 2:
        if data_valid.shape[-1] == 1:
            data_valid = np.squeeze(data_valid, axis=(-1,))

    target_train = np.asarray(target_train)
    target_valid = np.asarray(target_valid)

    validation_data = [data_valid, target_valid]





    print("Generating model..")

    EPOCHS = 15
    hiddens = [ [1000], [500], [200] ]
    #hiddens = [ [2000,1000], ]

    lrs = [10]
    batch_sizes = [32]
    optimizers_str = ['Adadelta']
    optimizers = [Adadelta]

    #hiddens = [ [1000], [2000], [4000], [2000,1000], [4000, 2000], [4000, 2000, 1000]]
    #lrs = [10, 1]
    #batch_sizes = [64, 32, 16]
    #optimizers_str = ['Adadelta', 'Adagrad']
    #optimizers = [Adadelta, Adagrad]

    for hid in hiddens:
        for opt, opt_str in zip(optimizers, optimizers_str):
            for lr in lrs:
                for bs in batch_sizes:

                    print ""
                    print("Training model..")
                    print "hiddens: " + str(hid)
                    print "optim:   " + str(opt_str)
                    print "lr:      " + str(lr)

                    fname = "im2docvec_opt-{}_lr-{}_bs-{}".format(opt_str, lr, bs)
                    for i, hu in enumerate(hid):
                        fname += "_hl-" + str(hu)
                    folder = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, fname)
                    if not os.path.isdir(folder):
                        os.mkdir(folder)

                    fname = os.path.join(folder, fname)

                    model = get_model(data_train.shape[1], target_train.shape[-1], hid)
                    model.compile(optimizer=opt(lr=lr), loss=cos_distance)

                    earlystop = EarlyStopping(monitor=MONITOR, min_delta=0.0005, patience=9)
                    reduceLR = ReduceLROnPlateau(monitor=MONITOR, factor=0.1, patience=4, verbose=1, epsilon=0.0005)
                    bestpoint = ModelCheckpoint(fname + '.model.best.h5', monitor=MONITOR, save_best_only=True)
                    checkpoint = ModelCheckpoint(fname + '.weights.{epoch:02d}.h5', monitor=MONITOR, save_best_only=False, save_weights_only=True)

                    #mAP_tr = ModelMAP(visual_features=visual_features, docs_vectors=text_features, class_list=class_list)
                    mAP_val = ModelMAP(visual_features=visual_features_valid, docs_vectors=text_features, class_list=class_list, history_key='val_mAP')
                    mAP_zs = ModelMAP(visual_features=visual_features_zs_test, docs_vectors=text_features_zs_test, class_list=class_list_test, history_key='zs_mAP')

                    callbacks = [reduceLR, bestpoint, checkpoint, mAP_val, mAP_zs] #, earlystop, ]
                    history = model.fit(data_train, target_train, batch_size=64, nb_epoch=EPOCHS, verbose=1, shuffle=True,
                                        callbacks=callbacks, validation_data=validation_data)

                    loss_csv = file(fname + '.loss.csv', 'w')
                    loss_csv.write('Epoch, Loss, Val Loss, valid mAP, test mAP\n')
                    epoch = 0
                    hist = history.history
                    for loss, val_loss, val_mAP, zs_mAP in zip(hist['loss'], hist['val_loss'], hist['val_mAP'], hist['zs_mAP'] ):
                        epoch+=1
                        loss_csv.write(str(epoch) + ', ' + str(loss) + ', ' + str(val_loss) + ', ' + str(val_mAP) + ', ' + str(zs_mAP) + '\n')




# def cos_distance(y_true, y_pred):
#     def l2_normalize(x, axis):
#         norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
#         return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
#     y_true = l2_normalize(y_true, axis=-1)
#     y_pred = l2_normalize(y_pred, axis=-1)
#     return -K.mean(y_true * y_pred, axis=-1)

def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def cosine_similarity_loss(y_true, y_pred):
    y_true_norm = K.l2_normalize(y_true, axis=-1)
    y_pred_norm = K.l2_normalize(y_pred, axis=-1)

    #numerator = K.sum(y_true * y_pred, axis=-2)
    #denominator = y_true_norm * y_pred_norm
    return -K.sum((y_true * y_pred) / (y_true_norm * y_pred_norm))

    #return K.sum(1 - ((y_true * y_pred) / (y_true_norm*y_pred_norm)))
    #return -K.mean(K.sum(y_true * y_pred) / (y_true_norm * y_pred_norm))



def get_model(input, output, dense=[2000]):
    model = Sequential()
    if dense is not None and len(dense) > 0:
        for i, d in enumerate(dense):
            if i == 0:
                model.add(Dense(input_dim=input, output_dim=d, activation='relu'))
            else:
                model.add(Dense(output_dim=d, activation='relu'))
        model.add(Dense(output_dim=output, activation=None))  # relu or sigmoid?
    else:
        model.add(Dense(input_dim=input, output_dim=output, activation=None))  # relu or sigmoid?
    print model.summary()
    return model


if __name__ == "__main__":
    main(sys.argv)
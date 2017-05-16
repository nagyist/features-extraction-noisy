import numpy as np
import codecs
import os


os.environ['KERAS_BACKEND'] = "tensorflow"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


from config import cfg
from keras.optimizers import Adadelta, sgd
from E6_joint_model.JointEmbedding import JointEmbedder
from E6_joint_model.mAP_callback import ModelMAP
from imdataset import ImageDataset




def main():

    cfg.init()

    class_list_500 = load_class_list('class_keep_from_pruning.txt', int_cast=True)
    class_list_400 = load_class_list('class_keep_from_pruning-train.txt', int_cast=True)
    class_list_100 = load_class_list('class_keep_from_pruning-test.txt', int_cast=True)

    visual_features_400_train = 'extracted_features/feat_dbp3120__resnet50__avg_pool_pruned-A@best_nb-classes-500_test-on-dbp3120__train.h5'
    visual_features_400_valid = 'extracted_features/feat_dbp3120__resnet50__avg_pool_pruned-A@best_nb-classes-500_test-on-dbp3120__valid.h5'
    visual_features_100_zs_test = 'extracted_features/feat_dbp3120__resnet50__avg_pool_pruned-A@best_nb-classes-500_test-on-dbp3120__test.h5'

    print("Loading textual features..")
    text_features_400 = np.load('docvec_400_train_on_wiki.npy')
    text_features_500 = np.load('docvec_500_train_on_wiki.npy')
    text_features_100_zs = []
    for i, cls in enumerate(class_list_500):
        if cls in class_list_100:
            text_features_100_zs.append(text_features_500[i])
    text_features_100_zs = np.asarray(text_features_100_zs)


    print("Loading visual features..")
    visual_features_400_train = ImageDataset().load_hdf5(visual_features_400_train)
    visual_features_100_zs_test = ImageDataset().load_hdf5(visual_features_100_zs_test)
    if visual_features_400_valid is not None:
        visual_features_400_valid = ImageDataset().load_hdf5(visual_features_400_valid)

    im_data_train = []
    tx_data_train = []
    label_train = []
    if visual_features_400_valid is not None:
        im_data_valid = []
        tx_data_valid = []
        label_valid = []

    for lbl, docv in zip(class_list_400, text_features_400):
        lbl = int(lbl)
        norm_docv = docv/np.linalg.norm(docv)  # l2 normalization

        visual_features_with_label = visual_features_400_train.sub_dataset_with_label(lbl)
        for visual_feat in visual_features_with_label.data:
            im_data_train.append(visual_feat)
            tx_data_train.append(norm_docv)
            label_train.append(lbl)

        if visual_features_400_valid is not None:
            visual_features_valid_with_label = visual_features_400_valid.sub_dataset_with_label(lbl)
            for visual_feat in visual_features_valid_with_label.data:
                im_data_valid.append(visual_feat)
                tx_data_valid.append(norm_docv)
                label_valid.append(lbl)



    # Image data conversion
    im_data_train = list_to_ndarray(im_data_train)
    im_data_valid = list_to_ndarray(im_data_valid)

    # Text data conversion
    tx_data_train = list_to_ndarray(tx_data_train)
    tx_data_valid = list_to_ndarray(tx_data_valid)

    # Label conversion
    label_train = list_to_ndarray(label_train)
    label_valid = list_to_ndarray(label_valid)


    joint_space_dim = 200
    batch_size = 32
    epochs = 40
    optimizer_str = 'Adadelta' # 'sgd'
    optimizer = Adadelta #sgd
    lr = 10


    print("Generating model..")
    print ""
    print("Training model..")
    print "optim:   " + str(optimizer_str)
    print "lr:      " + str(lr)

    fname = "jointmodel_opt-{}_lr-{}_bs-{}".format(optimizer_str, lr, batch_size)

    JE = JointEmbedder(im_dim=im_data_train.shape[-1], tx_dim=tx_data_train.shape[-1], out_dim=joint_space_dim,
                       n_text_classes=len(class_list_400))

    model = JE.model(optimizer=optimizer(lr=lr),
                     activation='sigmoid',
                     #tx_hidden_layers=[256], tx_hidden_activation=['softmax'],
                     #im_hidden_layers=[512], im_hidden_activation=['tanh'],
                     )

    # earlystop = EarlyStopping(monitor=MONITOR, min_delta=0.0005, patience=9)
    # reduceLR = ReduceLROnPlateau(monitor=MONITOR, factor=0.1, patience=4, verbose=1, epsilon=0.0005)
    # bestpoint = ModelCheckpoint(fname + '.model.best.h5', monitor=MONITOR, save_best_only=True)
    # checkpoint = ModelCheckpoint(fname + '.weights.{epoch:02d}.h5', monitor=MONITOR, save_best_only=False, save_weights_only=True)

    # # mAP_tr = ModelMAP(visual_features=visual_features, docs_vectors=text_features, class_list=class_list)
    mAP_tr = ModelMAP(visual_features=visual_features_400_train, docs_vectors=text_features_400,
                      class_list=class_list_400, history_key='tr_mAP',
                      exe_on_train_begin=False, on_train_begin_key='tr_begin-tr_map',
                      exe_on_batch_end=False, on_batch_end_key='batch-tr_map',
                      exe_on_epoch_end=False,
                      exe_on_train_end=True)
    mAP_val = ModelMAP(visual_features=visual_features_400_valid, docs_vectors=text_features_400,
                       class_list=class_list_400, history_key='val_mAP',
                       exe_on_train_begin=False, on_train_begin_key='tr_begin-val_map',
                       exe_on_batch_end=False, on_batch_end_key='batch-val_map',
                       exe_on_train_end=True, exe_on_epoch_end=False
                       )

    mAP_zs = ModelMAP(visual_features=visual_features_100_zs_test, docs_vectors=text_features_100_zs,
                      class_list=class_list_100, history_key='zs_mAP',
                      exe_on_train_begin=False, on_train_begin_key='tr_begin-zs_map',
                      exe_on_batch_end=False, on_batch_end_key='batch-zs_map',
                      exe_on_train_end=True, exe_on_epoch_end=False)

    # callbacks = [reduceLR, bestpoint, checkpoint, mAP_val, mAP_zs] #, earlystop, ]
    callbacks = [mAP_tr, mAP_val, mAP_zs]  # , mAP_val,  mAP_zs] #, earlystop, ]

    # label_train = np.zeros([len(im_data_train), 1])
    # label_valid = np.zeros([len(im_data_valid), 1])

    label_map = {}
    for index, label in enumerate(class_list_400):
        label_map[label] = index
    size = len(class_list_400)

    label_train_converted = []
    for l in label_train:
        new_l = np.zeros([size])
        new_l[label_map[l]] = 1
        label_train_converted.append(new_l)
    label_train_converted = np.asarray(label_train_converted)
    label_valid_converted = []
    for l in label_valid:
        new_l = np.zeros([size])
        new_l[label_map[l]] = 1
        label_valid_converted.append(new_l)
    label_valid_converted = np.asarray(label_valid_converted)
    #label_train_converted = np.asarray([label_map[l] for l in label_train])
    #label_valid_converted = np.asarray([label_map[l] for l in label_valid])


    model.summary()
    history = model.fit([im_data_train, tx_data_train], [label_train, label_train, label_train_converted],
                        validation_data=[[im_data_valid, tx_data_valid], [label_valid, label_valid, label_valid_converted]],
                        batch_size=batch_size, nb_epoch=epochs, shuffle=True,
                        verbose=1, callbacks=callbacks)

    loss_csv = file(fname + '.loss.csv', 'w')
    hist = history.history






def load_class_list(path,  encoding='utf-8', use_numpy=False, int_cast=False):
    ret = None
    if path is not None:
        ret = codecs.open(path, 'r', encoding=encoding).read().split('\n')
        if use_numpy:
            ret = np.asarray(ret, dtype='int32')
        elif int_cast:
            ret = map(int, ret)
    return ret

def save_class_list(class_list, path, encoding='utf-8'):
    # type: (list, basestring) -> None
    f = codecs.open(path, 'w', encoding=encoding)
    for cls in class_list:
        f.write(unicode(cls) + '\n')
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.close()

def list_to_ndarray(list, remove_final_empty_dims=True, min_dims=2):
    data = np.asarray(list)
    if remove_final_empty_dims:
        while len(data.shape) > min_dims:
            if data.shape[-1] == 1:
                data = np.squeeze(data, axis=(-1,))
            else:
                break
    return data
if __name__ == "__main__":
    main()

import os
# floatX=float32,device=gpu1,lib.cnmem=1,
# os.environ['THEANO_FLAGS'] = "exception_verbosity=high, optimizer=None"
from keras.callbacks import TensorBoard

os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# FORCE TENSORFLOW BACKEND HERE!! WITH THEANO THIS EXPERIMENT WON'T RUN!!
# NB: The problem of low performance of tensorflow backend with theano ordering is true only for convolutional layers!
#     We don't have to worry about that here.


import sys

from keras.optimizers import Adadelta

from E5_embedding import cfg_emb
from mAP_callback import ModelMAP
from E6_joint_model import JointEmbedding
from E6_joint_model.JointEmbedding import JointEmbedder
from imdataset import ImageDataset
import numpy as np

def main(args):
    joint_embedding_train()

def joint_embedding_train(visual_features=cfg_emb.VISUAL_FEATURES_TRAIN,
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

    print("Generating model..")

    EPOCHS = 50
    joint_space_dimensions = [ 200 ]
    #hiddens = [ [1000], [500], [200] ]
    hiddens = [ [200] ]

    lrs = [1]
    batch_sizes = [32]
    optimizers_str = ['Adadelta']
    optimizers = [Adadelta]

    #hiddens = [ [1000], [2000], [4000], [2000,1000], [4000, 2000], [4000, 2000, 1000]]
    #lrs = [10, 1]
    #batch_sizes = [64, 32, 16]
    #optimizers_str = ['Adadelta', 'Adagrad']
    #optimizers = [Adadelta, Adagrad]
    for joint_space_dim in joint_space_dimensions:
        for hid in hiddens:
            for opt, opt_str in zip(optimizers, optimizers_str):
                for lr in lrs:
                    for bs in batch_sizes:

                        print ""
                        print("Training model..")
                        print "hiddens: " + str(hid)
                        print "optim:   " + str(opt_str)
                        print "lr:      " + str(lr)

                        fname = "jointmodel_opt-{}_lr-{}_bs-{}".format(opt_str, lr, bs)
                        for i, hu in enumerate(hid):
                            fname += "_hl-" + str(hu)
                        folder = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, fname)
                        if not os.path.isdir(folder):
                            os.mkdir(folder)

                        fname = os.path.join(folder, fname)

                        JE = JointEmbedder(im_dim=im_data_train.shape[-1], tx_dim=tx_data_train.shape[-1],
                                           out_dim=joint_space_dim, )

                        model = JE.model(optimizer=opt(lr=lr),
                                         #activation='softmax',
                                         #tx_hidden_layers=[250], tx_hidden_activation=['relu'],
                                         #im_hidden_layers=[500], im_hidden_activation=['tanh'],
                                         )

                        #earlystop = EarlyStopping(monitor=MONITOR, min_delta=0.0005, patience=9)
                        #reduceLR = ReduceLROnPlateau(monitor=MONITOR, factor=0.1, patience=4, verbose=1, epsilon=0.0005)
                        #bestpoint = ModelCheckpoint(fname + '.model.best.h5', monitor=MONITOR, save_best_only=True)
                        #checkpoint = ModelCheckpoint(fname + '.weights.{epoch:02d}.h5', monitor=MONITOR, save_best_only=False, save_weights_only=True)

                        # # mAP_tr = ModelMAP(visual_features=visual_features, docs_vectors=text_features, class_list=class_list)
                        mAP_val = ModelMAP(visual_features=visual_features_valid, docs_vectors=text_features,
                                           class_list=class_list, history_key='val_mAP',
                                           exe_on_train_begin=False, on_train_begin_key='tr_begin-val_map',
                                           exe_on_batch_end=False, on_batch_end_key='batch-val_map')
                        mAP_tr  = ModelMAP(visual_features=visual_features, docs_vectors=text_features,
                                           class_list=class_list, history_key='tr_mAP',
                                           exe_on_train_begin=False, on_train_begin_key='tr_begin-tr_map',
                                           exe_on_batch_end=False, on_batch_end_key='batch-tr_map',
                                           exe_on_epoch_end=True,
                                           exe_on_train_end=False)
                        mAP_zs = ModelMAP(visual_features=visual_features_zs_test, docs_vectors=text_features_zs_test,
                                          class_list=class_list_test, history_key='zs_mAP',
                                          exe_on_train_begin=False, on_train_begin_key='tr_begin-zs_map',
                                          exe_on_batch_end=False, on_batch_end_key='batch-zs_map')


                        #callbacks = [reduceLR, bestpoint, checkpoint, mAP_val, mAP_zs] #, earlystop, ]
                        callbacks = [mAP_tr]#, mAP_val,  mAP_zs] #, earlystop, ]

                        # label_train = np.zeros([len(im_data_train), 1])
                        # label_valid = np.zeros([len(im_data_valid), 1])
                        model.summary()
                        history = model.fit([im_data_train, tx_data_train], [label_train, label_train, label_train],
                                            validation_data=[[im_data_valid, tx_data_valid], [label_valid, label_valid, label_valid]],
                                            batch_size=bs, nb_epoch=EPOCHS, shuffle=True,
                                            verbose=1, callbacks=callbacks)

                        loss_csv = file(fname + '.loss.csv', 'w')
                        hist = history.history


                        # if 'tr_begin-val_map' in hist.keys():
                        #     loss_csv.write('val_mAP pre train:, {}\n'.format(hist['tr_begin-val_map'][0]))
                        # if 'tr_begin-zs_map' in hist.keys():
                        #     loss_csv.write('zs_mAP pre train:, {}\n'.format(hist['tr_begin-zs_map'][0]))
                        #
                        # loss_csv.write('Epoch, Loss, Val Loss, valid mAP, test mAP\n')
                        # epoch = 0
                        # for loss, val_loss, val_mAP, zs_mAP in zip(hist['loss'], hist['val_loss'], hist['val_mAP'], hist['zs_mAP'] ):
                        #     epoch+=1
                        #     loss_csv.write(str(epoch) + ', ' + str(loss) + ', ' + str(val_loss) + ', ' + str(val_mAP) + ', ' + str(zs_mAP) + '\n')
                        #
                        # if 'batch-zs_map' in hist.keys() or 'batch-val_map' in hist.keys():
                        #     loss_csv.write('\n\n\n\nbatch_size:, {}\n\n'.format(bs))
                        #     loss_csv.write('Batch, val mAP, test mAP\n')
                        #     batch = 0
                        #     for val_mAP, zs_mAP in zip(hist['batch-val_map', 'batch-zs_map']):
                        #         batch += 1
                        #         loss_csv.write('{}, {}, {}\n'.format(batch, str(val_mAP), str(zs_mAP)))




if __name__ == "__main__":
    main(sys.argv)

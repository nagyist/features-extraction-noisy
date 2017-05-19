import os
# floatX=float32,device=gpu1,lib.cnmem=1,
# os.environ['THEANO_FLAGS'] = "exception_verbosity=high, optimizer=None"
import random

os.environ['KERAS_BACKEND'] = "tensorflow"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# FORCE TENSORFLOW BACKEND HERE!! WITH THEANO THIS EXPERIMENT WON'T RUN!!
# NB: The problem of low performance of tensorflow backend with theano ordering is true only for convolutional layers!
#     We don't have to worry about that here.



from keras.callbacks import TensorBoard, ModelCheckpoint
from theano.gof import optimizer



import sys

from keras.optimizers import Adadelta, Adam

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
    tx_data_train_im_aligned = []  # 1 text for each image (align: img_lbl_x <-> txt_lbl_x <-> lbl_x )
    tx_data_train = []  # 1 text for each class
    label_train = []
    if visual_features_valid is not None:
        im_data_valid = []
        tx_data_valid_im_aligned = []
        label_valid = []


    for lbl, docv in zip(cycle_clslst_txfeat[0], cycle_clslst_txfeat[1]):
        lbl = int(lbl)
        norm_docv = docv/np.linalg.norm(docv)  # l2 normalization
        tx_data_train.append(norm_docv)

        visual_features_with_label = visual_features.sub_dataset_with_label(lbl)
        for visual_feat in visual_features_with_label.data:
            visual_feat = visual_feat / np.linalg.norm(visual_feat)  # l2 normalization
            im_data_train.append(visual_feat)
            tx_data_train_im_aligned.append(norm_docv)
            label_train.append(lbl)

        if visual_features_valid is not None:
            visual_features_valid_with_label = visual_features_valid.sub_dataset_with_label(lbl)
            for visual_feat in visual_features_valid_with_label.data:
                visual_feat = visual_feat / np.linalg.norm(visual_feat)  # l2 normalization
                im_data_valid.append(visual_feat)
                tx_data_valid_im_aligned.append(norm_docv)
                label_valid.append(lbl)


    # Image data conversion
    im_data_train = list_to_ndarray(im_data_train)
    im_data_valid = list_to_ndarray(im_data_valid)

    # Text data conversion
    tx_data_train = list_to_ndarray(tx_data_train)
    tx_data_train_im_aligned = list_to_ndarray(tx_data_train_im_aligned)
    tx_data_valid_im_aligned = list_to_ndarray(tx_data_valid_im_aligned)


    # Label conversion
    label_train = list_to_ndarray(label_train)
    label_valid = list_to_ndarray(label_valid)


    print("Generating model..")

    MONITOR = 'val_loss'

    class Config:
        def __init__(self):
            self.lr = 10
            self.bs = 64
            self.epochs = 50
            self.opt = Adam
            self.opt_str= 'adam'
            self.joint_space_dim = 200
            self.tx_activation = 'softmax'
            self.im_activation = 'tanh'
            self.tx_hidden_layers = None
            self.tx_hidden_activation = None
            self.im_hidden_layers = None
            self.im_hidden_activation = None
            self.contrastive_loss_weight = 1
            self.logistic_loss_weight = 1
            self.contrastive_loss_weight_inverted=1
            self.weight_init='glorot_uniform'

    # GOT GREAT RESUTLS WITH THIS PARAMS:
    # configs = []
    # c = Config()
    # c.lr = 100
    # c.bs = 64
    # c.epochs = 50
    # c.joint_space_dim = 200
    # c.emb_activation = 'softmax'
    # c.contrastive_loss_weight = 3
    # c.logistic_loss_weight = 1
    # c.weight_init = 'glorot_uniform' # 'glorot_normal'
    #
    # # train_mAP-fit-end: 0.231570111798
    # # valid_mAP-fit-end: 0.36824232778
    # # test_mAP-fit-end: 0.12500124832
    # Epoch 48 / 50
    # loss: 2.8842 - activation_1_loss: 0.7106 - activation_2_loss: 0.7106 - dense_1_loss: 0.7524 - val_loss: 3.0216 - val_activation_1_loss: 0.8354 - val_activation_2_loss: 0.8354 - val_dense_1_loss: 0.5154
    # Epoch 49 / 50
    # loss: 2.7934 - activation_1_loss: 0.6958 - activation_2_loss: 0.6958 - dense_1_loss: 0.7061 - val_loss: 2.6629 - val_activation_1_loss: 0.5755 - val_activation_2_loss: 0.5755 - val_dense_1_loss: 0.9365
    # Epoch 50 / 50
    # loss: 2.7774 - activation_1_loss: 0.6948 - activation_2_loss: 0.6948 - dense_1_loss: 0.6930 - val_loss: 2.7351 - val_activation_1_loss: 0.5661 - val_activation_2_loss: 0.5661 - val_dense_1_loss: 1.0367


    # configs = []
    # c = Config()
    # c.lr = 100
    # c.bs = 64
    # c.epochs = 50
    # c.joint_space_dim = 200
    # c.emb_activation = 'softmax'
    # c.contrastive_loss_weight = 3
    # c.logistic_loss_weight = 1
    # c.weight_init = 'glorot_uniform' # 'glorot_normal'
    # c.tx_hidden_layers = [250]
    # c.tx_hidden_activation = ['relu']
    # c.im_hidden_layers = [500]
    # c.im_hidden_activation = ['tanh']
    configs = []
    c = Config()
    c.lr = 10
    c.opt = Adadelta
    c.opt_str = 'adadelta'
    c.bs = 64
    c.epochs = 30
    c.joint_space_dim = 200
    c.tx_activation = 'sigmoid'
    c.im_activation = 'sigmoid'
    c.contrastive_loss_weight = 1
    c.contrastive_loss_weight_inverted = 1
    c.logistic_loss_weight = 0
    c.weight_init = 'glorot_normal'
    #c.weight_init = 'glorot_uniform' # 'glorot_normal'
    #c.tx_hidden_layers = [200]
    #c.tx_hidden_activation = ['relu']
    #c.im_hidden_layers = [800]
    #c.im_hidden_activation = ['relu']
    # train_mAP-fit-end: 0.501253132832
    # valid_mAP-fit-end: 0.501253132832
    # test_mAP-fit-end: 0.505
    # # ... in realta' abbiamo tutti i vettori delle distanze IDENTICI per questo si hanno questi risultati

    configs.append(c)

    for c in configs:

                    print ""
                    print("Training model..")
                    print "optim:   " + str(c.opt_str)
                    print "lr:      " + str(c.lr)

                    #fname = "jointmodel_opt-{}_lr-{}_bs-{}".format(c.opt_str, c.lr, c.bs)
                    fname = "jointmodel"

                    # for i, hu in enumerate(hid):
                    #     fname += "_hl-" + str(hu)
                    folder = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, fname)
                    if not os.path.isdir(folder):
                        os.mkdir(folder)

                    fname = os.path.join(folder, fname)

                    JE = JointEmbedder(im_dim=im_data_train.shape[-1],
                                       tx_dim=tx_data_train.shape[-1],
                                       out_dim=c.joint_space_dim,
                                       n_text_classes=len(class_list)
                                       )

                    model = JE.model(optimizer=c.opt(lr=c.lr),
                                     tx_activation=c.tx_activation,
                                     im_activation=c.im_activation,
                                     tx_hidden_layers=c.tx_hidden_layers, tx_hidden_activation=c.tx_hidden_activation,
                                     im_hidden_layers=c.im_hidden_layers, im_hidden_activation=c.im_hidden_activation,
                                     contrastive_loss_weight=c.contrastive_loss_weight,
                                     logistic_loss_weight=c.logistic_loss_weight,
                                     contrastive_loss_weight_inverted=c.contrastive_loss_weight_inverted,
                                     init=c.weight_init,
                                     )

                    #earlystop = EarlyStopping(monitor=MONITOR, min_delta=0.0005, patience=9)
                    #reduceLR = ReduceLROnPlateau(monitor=MONITOR, factor=0.1, patience=4, verbose=1, epsilon=0.0005)
                    bestpoint = ModelCheckpoint(fname + '.model.best.h5', monitor=MONITOR, save_best_only=True)
                    checkpoint = ModelCheckpoint(fname + '.weights.{epoch:02d}.h5', monitor=MONITOR, save_best_only=False, save_weights_only=True)

                    mAP_tr = ModelMAP(visual_features=visual_features,
                                      docs_vectors=text_features,
                                      class_list=class_list,
                                      data_name='train-set',
                                      exe_fit_end=True,
                                      recall_at_k=[10]
                                      )
                    mAP_val = ModelMAP(visual_features=visual_features_valid,
                                       docs_vectors=text_features,
                                       class_list=class_list,
                                       data_name='valid-set',
                                       exe_fit_end=True,
                                       recall_at_k=[10]
                                       )
                    mAP_zs = ModelMAP(visual_features=visual_features_zs_test,
                                      docs_vectors=text_features_zs_test,
                                      class_list=class_list_test,
                                      data_name='test-set',
                                      exe_fit_end=True,
                                      recall_at_k=[10])


                    callbacks = [mAP_tr, mAP_val,  mAP_zs, checkpoint, bestpoint] #, earlystop, ]


                    model.summary()


                    label_map = {}
                    for index, label in enumerate(class_list):
                        label_map[label] = index
                    size = len(class_list)

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
                    # label_train_converted = np.asarray([label_map[l] for l in label_train])
                    # label_valid_converted = np.asarray([label_map[l] for l in label_valid])



                    # Creating contrastive training set:
                    im_contr_data_valid, tx_contr_data_valid, label_contr_valid, label_logistic_valid = \
                        create_contrastive_dataset(im_data_valid, tx_data_train, label_valid, class_list)


                    # # for ep in range(0, c.epochs):
                    #
                    # # Creating contrastive training set:
                    # im_contr_data_train, tx_contr_data_train, label_contr_train, label_logistic_train = \
                    #     create_contrastive_dataset(im_data_train, tx_data_train, label_train, class_list)
                    #
                    #
                    # history = model.fit([im_contr_data_train, tx_contr_data_train],
                    #                     [label_contr_train, label_contr_train, label_contr_train, label_logistic_train],
                    #                     validation_data=[[im_contr_data_valid, tx_contr_data_valid],
                    #                                      [label_contr_valid, label_contr_valid, label_contr_valid, label_logistic_valid]],
                    #                     batch_size=c.bs, nb_epoch=c.epochs, shuffle=True,
                    #                     verbose=1, callbacks=callbacks)

                    for ep in range(0, c.epochs):
                        bestpoint = ModelCheckpoint(fname + '.model.best.h5', monitor=MONITOR, save_best_only=True)
                        checkpoint = ModelCheckpoint(fname + '.weights.{epoch:02d}.h5' % {'epoch': c.epochs}, monitor=MONITOR,
                                                     save_best_only=False, save_weights_only=True)
                        callbacks = [checkpoint, bestpoint]  # , earlystop, ]
                        if ep == c.epochs-1:
                            callbacks = [mAP_tr, mAP_val, mAP_zs, checkpoint, bestpoint]  # , earlystop, ]
                        # Creating contrastive training set:
                        print("Epoch: " + str(ep))
                        im_contr_data_train, tx_contr_data_train, label_contr_train, label_logistic_train = \
                            create_contrastive_dataset(im_data_train, tx_data_train, label_train, class_list)

                        history = model.fit([im_contr_data_train, tx_contr_data_train],
                                            [label_contr_train, label_contr_train, label_logistic_train],
                                            validation_data=[[im_contr_data_valid, tx_contr_data_valid],
                                                             [label_contr_valid, label_contr_valid,
                                                              label_logistic_valid]],
                                            batch_size=c.bs, nb_epoch=1, shuffle=True,
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




def create_contrastive_dataset(im_data_train, tx_data_train, label_train, class_list):
    im_contr_data_train = []
    tx_contr_data_train = []
    label_contr_train = []
    label_logistic_train = []
    size = len(class_list)

    class_list_reverse = {lbl: index for index, lbl in enumerate(class_list)}
    for im, label in zip(im_data_train, label_train):
        contr_label = label
        while contr_label == label:
            contr_label = random.choice(class_list)

        label_index = class_list_reverse[label]
        contr_label_index = class_list_reverse[contr_label]

        # Correct example:
        tx = tx_data_train[label_index]
        im_contr_data_train.append(im)
        tx_contr_data_train.append(tx)
        label_contr_train.append(1)

        label_logistic_ok = np.zeros([size])
        label_logistic_ok[label_index] = 1
        label_logistic_train.append(label_logistic_ok)


        # Bad example:
        tx_bad = tx_data_train[contr_label_index]
        im_contr_data_train.append(im)
        tx_contr_data_train.append(tx_bad)
        label_contr_train.append(0)

        label_logistic_bad = np.zeros([size])
        label_logistic_bad[contr_label_index] = 1
        label_logistic_train.append(label_logistic_bad)


    return np.asarray(im_contr_data_train), np.asarray(tx_contr_data_train), np.asarray(label_contr_train), \
           np.asarray(label_logistic_train)



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

import os
from pprint import pprint
os.environ['KERAS_BACKEND'] = "tensorflow"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# FORCE TENSORFLOW BACKEND HERE!! WITH THEANO THIS EXPERIMENT WON'T RUN!!
# NB: The problem of low performance of tensorflow backend with theano ordering is true only for convolutional layers!
#     We don't have to worry about that here.

# floatX=float32,device=gpu1,lib.cnmem=1,
# os.environ['THEANO_FLAGS'] = "exception_verbosity=high, optimizer=None"

import random
from ExecutionConfig import Config
from config_generator import config_gen_TEST, config_gen_ACTIVATIONS, config_gen_DIMS
from keras.callbacks import ModelCheckpoint
from E5_embedding import cfg_emb
from mAP_callback import ModelMAP
from JointEmbedding import JointEmbedder
from imdataset import ImageDataset
import numpy as np


EVAL_MAP=True
EVAL_INIT_MODEL_LOSS=True
EVAL_INIT_MODEL_MAP=True
JOINT_PREDICTION_FOLDER = 'joint_prediction'
JOINT_MODEL_FOLDER = "joint_model"



if not os.path.isdir(JOINT_MODEL_FOLDER):
    os.mkdir(JOINT_MODEL_FOLDER)
if not os.path.isdir(JOINT_PREDICTION_FOLDER):
    os.mkdir(JOINT_PREDICTION_FOLDER)







def main():
    #joint_embedding_train(config_gen_function=config_gen_TEST, debug_map_val=1)
    joint_embedding_train(config_gen_function=config_gen_ACTIVATIONS)
    joint_embedding_train(config_gen_function=config_gen_DIMS)


def joint_embedding_train(config_gen_function=config_gen_TEST, debug_map_val=None):

    visual_features = cfg_emb.VISUAL_FEATURES_TRAIN
    text_features = cfg_emb.TEXT_FEATURES_400
    class_list = cfg_emb.CLASS_LIST_400
    visual_features_valid = cfg_emb.VISUAL_FEATURES_VALID
    visual_features_zs_test = cfg_emb.VISUAL_FEATURES_TEST
    text_features_zs_test = cfg_emb.TEXT_FEATURES_100
    class_list_test = cfg_emb.CLASS_LIST_100
    recall_at_k = [1, 3, 5, 10]

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
        im_data_val = []
        tx_data_valid_im_aligned = []
        label_val = []


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
                im_data_val.append(visual_feat)
                tx_data_valid_im_aligned.append(norm_docv)
                label_val.append(lbl)


    # Image data conversion
    im_data_train = list_to_ndarray(im_data_train)
    im_data_val = list_to_ndarray(im_data_val)

    # Text data conversion
    tx_data_train = list_to_ndarray(tx_data_train)
    #tx_data_train_im_aligned = list_to_ndarray(tx_data_train_im_aligned)
    #tx_data_valid_im_aligned = list_to_ndarray(tx_data_valid_im_aligned)


    # Label conversion
    label_train = list_to_ndarray(label_train)
    label_val = list_to_ndarray(label_val)


    print("Generating model..")


    configs, config_gen_name = config_gen_function()

    print("Executing training over config generator: " + config_gen_name)
    folder_gen_name = "jointmodel_confgen-" + config_gen_name
    folder_gen_path = os.path.join(JOINT_MODEL_FOLDER, folder_gen_name)
    if not os.path.isdir(folder_gen_path):
        os.mkdir(folder_gen_path)

    class ModelScore:
        def __init__(self, train_set_score=None, valid_set_score=None,  test_set_score=None):
            self.train_set = train_set_score
            self.valid_set = valid_set_score
            self.test_set = test_set_score

    class ConfigScore:
        def __init__(self, name=None):
            self.name=name
            self.scores_best_train = ModelScore()
            self.scores_best_valid = ModelScore()
            self.scores_init = ModelScore()



    config_scores = [] # list of Score, one for each config

    for config_counter, c in enumerate(configs):
        if not isinstance(c, Config):
            raise TypeError('c is not an instance of Config class.')

        print("")
        print("")
        print("")
        print("")
        print("Config: ")
        pprint(c)


        fname = folder_gen_name + "__" + str(config_counter)
        folder_path = os.path.join(folder_gen_path, fname)
        fpath = os.path.join(folder_path, fname)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        c.saveJSON(fpath + '.config.json')

        JE = JointEmbedder(im_dim=im_data_train.shape[-1], tx_dim=tx_data_train.shape[-1], out_dim=c.sp_dim, n_text_classes=len(class_list))

        model = JE.model(optimizer=c.opt(lr=c.lr),
                         tx_activation=c.tx_act,
                         im_activation=c.im_act,
                         tx_hidden_layers=c.tx_hid,
                         im_hidden_layers=c.im_hid,
                         contrastive_loss_weight=c.contr_w,
                         logistic_loss_weight=c.log_w_im,
                         contrastive_loss_weight_inverted=c.contr_inv_w,
                         init=c.w_init)

        model.summary()


        label_map = {}
        for index, label in enumerate(class_list):
            label_map[label] = index
        size = len(class_list)

        # label_train_converted = []
        # for l in label_train:
        #     new_l = np.zeros([size])
        #     new_l[label_map[l]] = 1
        #     label_train_converted.append(new_l)
        # label_train_converted = np.asarray(label_train_converted)
        # label_valid_converted = []
        # for l in label_valid:
        #     new_l = np.zeros([size])
        #     new_l[label_map[l]] = 1
        #     label_valid_converted.append(new_l)
        # label_valid_converted = np.asarray(label_valid_converted)
        # label_train_converted = np.asarray([label_map[l] for l in label_train])
        # label_valid_converted = np.asarray([label_map[l] for l in label_valid])




        init_model_fname = fpath + '.model.init.random.h5'
        best_valid_fname = fpath + '.model.best.val_loss.h5'
        best_train_fname = fpath + '.model.best.loss.h5'
        model.save(init_model_fname)

        # Creating contrastive training set:
        val_x_im, val_x_tx, val_y_contr, val_y_log = get_contr_data(im_data_val, tx_data_train, label_val, class_list)
        val_X = [val_x_im, val_x_tx]
        val_Y = [val_y_contr, val_y_contr, val_y_log]

        best_loss = best_val_loss = float('inf')
        best_loss_epoch = -1
        best_val_loss_epoch = -1
        loss_hist = []
        val_loss_hist = []

        for ep in range(0, c.epochs):
            print("Epoch: {}/{}".format(ep, c.epochs-1))

            checpoint_path = fpath + ".weights.{:03d}.h5".format(ep)
            checkpoint = ModelCheckpoint(checpoint_path, monitor='val_loss', save_best_only=False, save_weights_only=True)

            x_im, x_tx, y_cont, y_log = get_contr_data(im_data_train, tx_data_train, label_train, class_list)
            X = [x_im, x_tx]
            Y = [y_cont, y_cont, y_log]

            hs = model.fit(X, Y, c.bs, nb_epoch=1, validation_data=[val_X, val_Y],shuffle=True, callbacks=[checkpoint])

            hist = hs.history
            val_loss = hist['val_loss'][0]
            loss = hist['loss'][0]
            val_loss_hist.append(val_loss)
            loss_hist.append(loss)

            if loss<best_loss:
                best_loss = loss
                model.save(best_train_fname)
                best_loss_epoch = ep
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                model.save(best_valid_fname)
                best_val_loss_epoch = ep





        loss_csv = file(fpath + ".loss.csv", 'w')
        loss_csv.write('Learning curves (loss),Epoch, Loss, Val Loss\n')

        if EVAL_INIT_MODEL_LOSS:
            x_im, x_tx, y_cont, y_log = get_contr_data(im_data_train, tx_data_train, label_train, class_list)
            X = [x_im, x_tx]
            Y = [y_cont, y_cont, y_log]
            init_loss = model.evaluate(X, Y, batch_size=c.bs)[0]
            init_val_loss = model.evaluate(val_X, val_Y, batch_size=c.bs)[0]
            loss_csv.write(', {}, {}, {}\n'.format(-1, init_loss, init_val_loss))

        epoch = 0
        for loss, val_loss in zip(loss_hist, val_loss_hist):
            loss_csv.write(", {}, {}, {}\n".format(epoch, loss, val_loss))
            epoch+=1
        loss_csv.write("\n\n\n")
        loss_csv.write("Best loss epoch:, {}, \n".format(best_loss_epoch))
        loss_csv.write("Best val loss epoch:, {}\n".format(best_val_loss_epoch))


        if EVAL_MAP:
            map_call_tr = ModelMAP(visual_features=visual_features, docs_vectors=text_features, class_list=class_list,
                                   data_name='TrainSet',
                                   text_retrieval_map=True, image_retrieval_map=True, recall_at_k=recall_at_k,
                                   debug_value=debug_map_val)

            map_call_val = ModelMAP(visual_features=visual_features_valid, docs_vectors=text_features, class_list=class_list,
                                    data_name='ValidSet',
                                    text_retrieval_map=True, image_retrieval_map=True, recall_at_k=recall_at_k,
                                    debug_value=debug_map_val)

            map_call_zs = ModelMAP(visual_features=visual_features_zs_test, docs_vectors=text_features_zs_test, class_list=class_list_test,
                                   data_name='TestSetZS',
                                   text_retrieval_map=True, image_retrieval_map=True, recall_at_k=recall_at_k,
                                   debug_value=debug_map_val)

            # Map on best loss model
            best_train_model = JointEmbedder.load_model(best_train_fname)
            map_tr_best_train = map_call_tr.call_manual(best_train_model)
            map_val_best_train = map_call_val.call_manual(best_train_model)
            map_zs_best_train = map_call_zs.call_manual(best_train_model)

            score_best_train = ModelScore(map_tr_best_train, map_val_best_train, map_zs_best_train)

            # Map on best val_loss model
            best_valid_model = JointEmbedder.load_model(best_valid_fname)
            map_tr_best_valid = map_call_tr.call_manual(best_valid_model)
            map_val_best_valid = map_call_val.call_manual(best_valid_model)
            map_zs_best_valid = map_call_zs.call_manual(best_valid_model)

            score_best_valid = ModelScore(map_tr_best_valid, map_val_best_valid, map_zs_best_valid)


            list_map_labels = [ "Best Tr Loss", "Best Val Loss"]
            list_map_dict_tr = [ map_tr_best_train, map_tr_best_valid]
            list_map_dict_val = [map_val_best_train, map_val_best_valid]
            list_map_dict_zs = [map_zs_best_train, map_zs_best_valid]

            score_init = None

            if EVAL_INIT_MODEL_MAP:
                # Map on init/random model
                init_model = JointEmbedder.load_model(init_model_fname)
                map_tr_init = map_call_tr.call_manual(init_model)
                map_val_init = map_call_val.call_manual(init_model)
                map_zs_init = map_call_zs.call_manual(init_model)
                list_map_labels.append("Init/Random")
                list_map_dict_tr.append(map_tr_init)
                list_map_dict_val.append(map_val_init)
                list_map_dict_zs.append(map_zs_init)

                score_init = ModelScore(map_tr_init, map_val_init, map_zs_init)

            cs = ConfigScore(name=str(config_counter))
            cs.scores_best_train = score_best_train
            cs.scores_best_valid = score_best_valid
            cs.scores_init = score_init
            config_scores.append(cs)


            loss_csv.write("\n\n\n\n")

            loss_csv.write(", Loaded models/weights:, ")
            for l in list_map_labels:
                loss_csv.write("{}, ".format(l))
            loss_csv.write("\n")

            loss_csv.write("\nmAP over training set, ")
            for key in map_tr_best_train.keys():
                loss_csv.write("{}, ".format(key))
                for map_dict in list_map_dict_tr:
                    loss_csv.write("{}, ".format(map_dict[key]))
                loss_csv.write("\n, ")

            loss_csv.write("\nmAP over validation set, ")
            for key in map_tr_best_train.keys():
                loss_csv.write("{}, ".format(key))
                for map_dict in list_map_dict_val:
                    loss_csv.write("{}, ".format(map_dict[key]))
                loss_csv.write("\n, ")

            loss_csv.write("\nmAP over zs-test set, ")
            for key in map_tr_best_train.keys():
                loss_csv.write("{}, ".format(key))
                for map_dict in list_map_dict_zs:
                    loss_csv.write("{}, ".format(map_dict[key]))
                loss_csv.write("\n, ")

        #
        # def write_map_dict(map_dict, str):
        #         loss_csv.write("\n" + str)
        #         for k, v in map_dict.items():
        #             loss_csv.write(",{}, {}\n".format(k, v))
        #
        #     write_map_dict(map_tr_best_train, "mAP on tr set - best loss model")
        #     write_map_dict(map_val_best_train, "mAP on val set - best loss model")
        #     write_map_dict(map_zs_best_train, "mAP on test zs set - best loss model")
        #     loss_csv.write("\n")
        #     write_map_dict(map_tr_best_valid, "mAP on tr set - best valid loss model")
        #     write_map_dict(map_val_best_valid, "mAP on val set - best valid loss model")
        #     write_map_dict(map_zs_best_valid, "mAP on test zs set - best valid loss model")
        #     loss_csv.write("\n")
        #     write_map_dict(map_tr_init, "mAP on tr set - init/random model")
        #     write_map_dict(map_val_init, "mAP on val set - init/random model")
        #     write_map_dict(map_zs_init, "mAP on test zs set - init/random model")

        loss_csv.close()



    if EVAL_MAP:

        assert cs.scores_best_train.test_set.keys() == \
               cs.scores_best_train.train_set.keys() == \
               cs.scores_best_train.valid_set.keys() == \
               cs.scores_best_valid.test_set.keys() == \
               cs.scores_best_valid.train_set.keys() == \
               cs.scores_best_valid.valid_set.keys()

        if EVAL_INIT_MODEL_MAP:
            assert cs.scores_best_train.test_set.keys() == \
                   cs.scores_init.test_set.keys() == \
                   cs.scores_init.train_set.keys() == \
                   cs.scores_init.valid_set.keys()

        keys = cs.scores_best_train.test_set.keys()
        for key in keys:

            stats_csv = file(os.path.join(folder_gen_path, folder_gen_name+".{}.csv".format(key)), 'w')
            stats_csv.write('Stats for {}\n\n'.format(key))

            init_model_comma = ', ' if EVAL_INIT_MODEL_MAP else ''
            stats_csv.write(', test over training set, , , , test over validation set, , , , test over test set, , ,, \n')

            stats_csv.write('Model Weights:, '
                            'best tr loss, best val loss, init/random, , '
                            'best tr loss, best val loss, init/random, , '
                            'best tr loss, best val loss, init/random, , \n')
            stats_csv.write('Config index/name, \n')

            for cs in config_scores:
                index = cs.name
                stats_csv.write('{}, {}, {}, {}, , {}, {}, {}, , {}, {}, {},\n'
                                .format(cs.name,
                                        cs.scores_best_train.train_set[key],
                                        cs.scores_best_valid.train_set[key],
                                        str(cs.scores_init.train_set[key]) if EVAL_INIT_MODEL_MAP else '',

                                        cs.scores_best_train.valid_set[key],
                                        cs.scores_best_valid.valid_set[key],
                                        str(cs.scores_init.valid_set[key]) if EVAL_INIT_MODEL_MAP else '',

                                        cs.scores_best_train.test_set[key],
                                        cs.scores_best_valid.test_set[key],
                                        str(cs.scores_init.test_set[key]) if EVAL_INIT_MODEL_MAP else ''
                                        ))

                # TODO: make a single csv for each KEY in:   configScore.scores_best_train.test_set['KEY']
                # TODO: where KEY is 'map-tx-..' 'map-im-..' 'recall@K'...
                # Each CSV have to show in each row a different configuration, in each column a different combination of:
                #   <model_used_to_compute_scores, dataset_used_to_compute_scores>
                # for example: <best_train, test_set> , <best_vaild, train_set>..
                #
                #
                # Final result should be like:
                #
                # CFG   best_train@tr_set   best_valid@tr_set   init@tr_set     |   best_train@valid_set    best_valid@valid_set    init@valid_set      |    best_train@test_set        best_valid@test_set     init@test_set
                #  0        0.244               0.xxx               ...
                #  1        0.xxx               0.xxx               ...
                #  2        0.xx                0.xxx               ...
                #  3        0.xxx               0.xxx               ...




def get_contr_data(im_data_train, tx_data_train, label_train, class_list):
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

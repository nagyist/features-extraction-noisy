import os

import pyprind
import scipy
from keras.engine import Model
from keras.models import load_model
from keras.layers import K
import numpy as np

from zzE6_joint_model.JointEmbedding import get_sub_model
from imdataset import ImageDataset

DEFAULT_JOINT_MODEL_EXT = '.model.best.h5'
JOINT_MODEL_FOLDER = "TODO"
JOINT_MODEL_PREDICTIONS_FOLDER = "TODO"
FEAT_TRAINSET_NAME = "TODO"

def joint_prediction_fname(im2doc_model_name, submodel, dataset_name=FEAT_TRAINSET_NAME):
    if submodel not in ['txt', 'img']:
        ValueError("submodel must be a string with value 'txt' or 'img'")
    return 'embedding_from_dataset-' + dataset_name + '_' + im2doc_model_name + '.' + submodel + '.npy' # TODO

def load_class_list(class_list_doc2vec):
    from E5_embedding.cfg_emb import load_class_list
    return load_class_list(class_list_doc2vec) # TODO


def retrieve_image_map(img_features, txt_features, class_list_doc2vec, joint_model,
                      joint_model_ext=None, joint_model_weights_ext=None, load_precomputed_embedded_feat=None,
                      verbose=False, progressbar=True):
    def printv(str):
        if verbose:
            print(str)

    if joint_model_ext is None:
        joint_model_ext = DEFAULT_JOINT_MODEL_EXT
    if load_precomputed_embedded_feat is None:
        load_precomputed_embedded_feat = False
    else:
        ValueError("load_precomputed_embedded_feat: not yet implemented.")

    printv("Loading visual features..")
    if not isinstance(img_features, ImageDataset):
        img_features = ImageDataset().load_hdf5(img_features)

    printv("Loading joint model..")
    if not isinstance(joint_model, Model):
        joint_model_name = joint_model
        model_file = os.path.join(JOINT_MODEL_FOLDER,
                                  os.path.join(joint_model_name, joint_model_name + joint_model_ext))
        joint_model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
    else:
        joint_model_name = None

    if joint_model_weights_ext is not None:
        printv("Loading joint model weights..")
        weight_file = os.path.join(JOINT_MODEL_FOLDER, os.path.join(joint_model, joint_model + joint_model_weights_ext))
        joint_model.load_weights(weight_file)



    if joint_model_name is not None:
        img_emb_path = os.path.join(JOINT_MODEL_PREDICTIONS_FOLDER, joint_prediction_fname(joint_model_name, 'img'))
        txt_emb_path = os.path.join(JOINT_MODEL_PREDICTIONS_FOLDER, joint_prediction_fname(joint_model_name, 'txt'))
    else:
        img_emb_path = "precomputed_im_emb.img.npy.temp"
        txt_emb_path = "precomputed_tx_emb.txt.npy.temp"

    if load_precomputed_embedded_feat and os.path.exists(img_emb_path)  and os.path.exists(txt_emb_path):
        printv("Pre computed embedding from images and text found... loading...")
        imgs_embedded = np.load(img_emb_path)
        txts_embedded = np.load(txt_emb_path)

    else:
        printv("Predict embedding from images and text(joint model embedding)...")

        img_data = img_features.data
        while len(img_data.shape) > 2:
            if img_data.shape[-1] == 1:
                img_data = np.squeeze(img_data, axis=(-1,))
        img_emb_model = get_sub_model(joint_model, 'img')
        imgs_embedded = img_emb_model.predict(img_data, verbose=verbose)
        np.save(img_emb_path, imgs_embedded)

        txt_data = txt_features
        while len(txt_data.shape) > 2:
            if txt_data.shape[-1] == 1:
                txt_data = np.squeeze(txt_data, axis=(-1,))
        txt_emb_model = get_sub_model(joint_model, 'txt')
        txts_embedded = txt_emb_model.predict(txt_data, verbose=verbose)
        np.save(txt_emb_path, txts_embedded)

        #[a, b, c] = joint_model.predict(x=[img_data[0:len(txt_data)], txt_data], verbose=verbose)

    if not isinstance(class_list_doc2vec, list):
        class_list_doc2vec = load_class_list(class_list_doc2vec)

    # mAP test (optimized with cdist)
    if progressbar:
        import sys
        bar = pyprind.ProgBar(len(txts_embedded), stream = sys.stdout)
    av_prec = []
    from scipy.spatial.distance import cdist

    #C = cdist(txts_embedded, imgs_embedded, 'cos')
    #C = 1-C
    C = -cdist(txts_embedded, imgs_embedded, 'cos')
    #C = (np.sqrt(C.shape[0])-C)/np.sqrt(C.shape[0])

    for i, dv in enumerate(txts_embedded):
        scores = []
        targets = []
        if progressbar:
            bar.update()

        lbl = int(class_list_doc2vec[i])
        for j, im_label in enumerate(img_features.labels):
            target = not bool(im_label[0] - lbl)
            score = C[i,j]
            scores.append(score)
            targets.append(target)

        from sklearn.metrics import average_precision_score
        AP = average_precision_score(targets, scores)
        av_prec.append(AP)
        printv("Class {} - AP = {}".format(lbl, AP))

    mAP = np.mean(np.asarray(av_prec))
    printv("\t\tmAP = {}".format(mAP))
    return mAP





def retrieve_text_map(img_features, txt_features, class_list_doc2vec,
                     joint_model,joint_model_ext=None, joint_model_weights_ext=None,
                     load_precomputed_embedded_feat=None, verbose=False, progressbar=True):
    def printv(str):
        if verbose:
            print(str)

    if joint_model_ext is None:
        joint_model_ext = DEFAULT_JOINT_MODEL_EXT
    if load_precomputed_embedded_feat is None:
        load_precomputed_embedded_feat = False
    else:
        ValueError("load_precomputed_embedded_feat: not yet implemented.")

    printv("Loading visual features..")
    if not isinstance(img_features, ImageDataset):
        img_features = ImageDataset().load_hdf5(img_features)

    printv("Loading im2doc model..")
    if not isinstance(joint_model, Model):
        joint_model_name = joint_model
        model_file = os.path.join(JOINT_MODEL_FOLDER,
                                  os.path.join(joint_model_name, joint_model_name + joint_model_ext))
        joint_model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
    else:
        joint_model_name = None

    if joint_model_weights_ext is not None:
        printv("Loading im2doc weights..")
        weight_file = os.path.join(JOINT_MODEL_FOLDER, os.path.join(joint_model, joint_model + joint_model_weights_ext))
        joint_model.load_weights(weight_file)



    if joint_model_name is not None:
        img_emb_path = os.path.join(JOINT_MODEL_PREDICTIONS_FOLDER, joint_prediction_fname(joint_model_name, 'img'))
        txt_emb_path = os.path.join(JOINT_MODEL_PREDICTIONS_FOLDER, joint_prediction_fname(joint_model_name, 'txt'))
    else:
        img_emb_path = "precomputed_im_emb.img.npy.temp"
        txt_emb_path = "precomputed_tx_emb.txt.npy.temp"

    if load_precomputed_embedded_feat and os.path.exists(img_emb_path)  and os.path.exists(txt_emb_path):
        printv("Pre computed embedding from images and text found... loading...")
        imgs_embedded = np.load(img_emb_path)
        txts_embedded = np.load(txt_emb_path)

    else:
        printv("Predict embedding from images and text(joint model embedding)...")

        img_data = img_features.data
        while len(img_data.shape) > 2:
            if img_data.shape[-1] == 1:
                img_data = np.squeeze(img_data, axis=(-1,))
        img_emb_model = get_sub_model(joint_model, 'img')
        imgs_embedded = img_emb_model.predict(img_data, verbose=verbose)
        np.save(img_emb_path, imgs_embedded)

        txt_data = txt_features
        while len(txt_data.shape) > 2:
            if txt_data.shape[-1] == 1:
                txt_data = np.squeeze(txt_data, axis=(-1,))
        txt_emb_model = get_sub_model(joint_model, 'txt')
        txts_embedded = txt_emb_model.predict(txt_data, verbose=verbose)
        np.save(txt_emb_path, txts_embedded)

        #[a, b, c] = joint_model.predict(x=[img_data[0:len(txt_data)], txt_data], verbose=verbose)

    if not isinstance(class_list_doc2vec, list):
        class_list_doc2vec = load_class_list(class_list_doc2vec)
    class_list_inverted_doc2vec = {k: i for i,k in enumerate(class_list_doc2vec)}
    # mAP test (optimized with cdist)
    if progressbar:
        import sys
        bar = pyprind.ProgBar(len(imgs_embedded), stream = sys.stdout)
    av_prec = []
    from scipy.spatial.distance import cdist

    #C = cdist(txts_embedded, imgs_embedded, 'cos')
    #C = 1-C
    C = -cdist(imgs_embedded, txts_embedded, 'euclidean')
    #C = (np.sqrt(C.shape[0])-C)/np.sqrt(C.shape[0])

    for i, iv in enumerate(imgs_embedded):
        scores = []
        #targets = []
        if progressbar:
            bar.update()

        lbl = int(img_features.labels[i])
        targets = np.zeros([txts_embedded.shape[0]])
        targets[int(class_list_inverted_doc2vec[lbl])] = 1

        for j, tx_vec in enumerate(txts_embedded):
            score = C[i,j]
            scores.append(score)

        from sklearn.metrics import average_precision_score
        AP = average_precision_score(targets, scores)
        av_prec.append(AP)
        printv("Img {} - AP = {}".format(lbl, AP))

    mAP = np.mean(np.asarray(av_prec))
    printv("\t\tmAP = {}".format(mAP))
    return mAP





def recall_top_k(img_features, txt_features, class_list_doc2vec,
                 joint_model, joint_model_ext=None, joint_model_weights_ext=None,
                 load_precomputed_embedded_feat=None,
                 top_k=10,
                 verbose=False, progressbar=True):
    def printv(str):
        if verbose:
            print(str)

    if joint_model_ext is None:
        joint_model_ext = DEFAULT_JOINT_MODEL_EXT
    if load_precomputed_embedded_feat is None:
        load_precomputed_embedded_feat = False
    else:
        ValueError("load_precomputed_embedded_feat: not yet implemented.")

    printv("Loading visual features..")
    if not isinstance(img_features, ImageDataset):
        img_features = ImageDataset().load_hdf5(img_features)

    printv("Loading im2doc model..")
    if not isinstance(joint_model, Model):
        joint_model_name = joint_model
        model_file = os.path.join(JOINT_MODEL_FOLDER,
                                  os.path.join(joint_model_name, joint_model_name + joint_model_ext))
        joint_model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
    else:
        joint_model_name = None

    if joint_model_weights_ext is not None:
        printv("Loading im2doc weights..")
        weight_file = os.path.join(JOINT_MODEL_FOLDER, os.path.join(joint_model, joint_model + joint_model_weights_ext))
        joint_model.load_weights(weight_file)



    if joint_model_name is not None:
        img_emb_path = os.path.join(JOINT_MODEL_PREDICTIONS_FOLDER, joint_prediction_fname(joint_model_name, 'img'))
        txt_emb_path = os.path.join(JOINT_MODEL_PREDICTIONS_FOLDER, joint_prediction_fname(joint_model_name, 'txt'))
    else:
        img_emb_path = "precomputed_im_emb.img.npy.temp"
        txt_emb_path = "precomputed_tx_emb.txt.npy.temp"

    if load_precomputed_embedded_feat and os.path.exists(img_emb_path)  and os.path.exists(txt_emb_path):
        printv("Pre computed embedding from images and text found... loading...")
        imgs_embedded = np.load(img_emb_path)
        txts_embedded = np.load(txt_emb_path)

    else:
        printv("Predict embedding from images and text(joint model embedding)...")

        img_data = img_features.data
        while len(img_data.shape) > 2:
            if img_data.shape[-1] == 1:
                img_data = np.squeeze(img_data, axis=(-1,))
        img_emb_model = get_sub_model(joint_model, 'img')
        imgs_embedded = img_emb_model.predict(img_data, verbose=verbose)
        np.save(img_emb_path, imgs_embedded)

        txt_data = txt_features
        while len(txt_data.shape) > 2:
            if txt_data.shape[-1] == 1:
                txt_data = np.squeeze(txt_data, axis=(-1,))
        txt_emb_model = get_sub_model(joint_model, 'txt')
        txts_embedded = txt_emb_model.predict(txt_data, verbose=verbose)
        np.save(txt_emb_path, txts_embedded)

        #[a, b, c] = joint_model.predict(x=[img_data[0:len(txt_data)], txt_data], verbose=verbose)

    if not isinstance(class_list_doc2vec, list):
        class_list_doc2vec = load_class_list(class_list_doc2vec)
    class_list_inverted_doc2vec = {k: i for i,k in enumerate(class_list_doc2vec)}
    # mAP test (optimized with cdist)
    if progressbar:
        import sys
        bar = pyprind.ProgBar(len(imgs_embedded), stream = sys.stdout)

    from scipy.spatial.distance import cdist

    #C = cdist(txts_embedded, imgs_embedded, 'cos')
    #C = 1-C
    C = -cdist(imgs_embedded, txts_embedded, 'euclidean')
    #C = (np.sqrt(C.shape[0])-C)/np.sqrt(C.shape[0])
    recall_per_img = []
    for i, iv in enumerate(imgs_embedded):
        if progressbar:
            bar.update()
        lbl = int(img_features.labels[i])
        arg_lbl = class_list_inverted_doc2vec[lbl]

        dists = C[i, :]
        arg_sort_dist = np.argsort(dists)

        if arg_lbl in arg_sort_dist[0:top_k+1]:
            recall_per_img.append(1)
        else:
            recall_per_img.append(0)


    return np.sum(recall_per_img)/float(len(recall_per_img))



def distances(src_vector, cmp_vectors, get_first_n=None):
    dists = []
    for vec in cmp_vectors:
        d = scipy.spatial.distance.cosine(src_vector, vec)
        dists.append(d)

    dists = np.asarray(dists)
    permutation = dists.argsort()
    dists = dists[permutation]
    return permutation[0:get_first_n], dists[0:get_first_n]


def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))




def progress(count, total, status='', pycharm_fix=True):
    import sys
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if pycharm_fix:
        sys.stdout.write('\r[%s] %s%s %s' % (bar, percents, '%', status))    # works on pycharm
    else:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))  # better for terminal, not working on pycharm
    sys.stdout.flush()

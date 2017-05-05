import numpy as np
import os

import pyprind
import scipy
from gensim import matutils
from keras.models import load_model, Model
from keras.layers import K

from E5_embedding import cfg_emb
from imdataset import ImageDataset


DEFAULT_IM2DOC_MODEL_EXT = '.model.best.h5'

def im2doc_prediction_fname(im2doc_model_name, im_dataset=cfg_emb.FEAT_TRAINSET_NAME):
    return 'docs_from_dataset-' + im_dataset  + '_' + im2doc_model_name + '.npy'

def test_embedding_top_similars(visual_features, docs_vectors_npy, class_list_doc2vec, im2doc_model_name,
                                im2doc_model_ext=None, im2doc_weights_ext=None,
                                load_precomputed_imdocs=None, top_similars=None):
    if im2doc_model_ext is None:
        im2doc_model_ext = DEFAULT_IM2DOC_MODEL_EXT
    if load_precomputed_imdocs is None:
        load_precomputed_imdocs = False
    if top_similars is None:
        top_similars = 10

    print("Loading visual features..")
    if not isinstance(visual_features, ImageDataset):
        visual_features = ImageDataset().load_hdf5(visual_features)


    print("Loading im2doc model..")
    model_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(im2doc_model_name, im2doc_model_name + im2doc_model_ext))
    im2doc_model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
    if im2doc_weights_ext is not None:
        print("Loading im2doc weights..")
        weight_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(im2doc_model, im2doc_model + im2doc_weights_ext))
        im2doc_model.load_weights(weight_file)


    imdocs_path = os.path.join(cfg_emb.IM2DOC_PREDICTION_FOLDER, im2doc_prediction_fname(im2doc_model_name))
    if load_precomputed_imdocs and os.path.exists(imdocs_path):
        print("Pre computed docs from images found (im2doc embedding)... loading...")
        output_doc_vectors = np.load(imdocs_path)
    else:
        print("Predict docs from images (im2doc embedding)..")
        im_data = visual_features.data
        while len(im_data.shape) > 2:
            if im_data.shape[-1] == 1:
                im_data = np.squeeze(im_data, axis=(-1,))
        output_doc_vectors = im2doc_model.predict(im_data, verbose=True)
        np.save(imdocs_path, output_doc_vectors)

    print("Loading doc2vec vectors...")
    if not isinstance(docs_vectors_npy, np.ndarray):
        docs_vectors_npy = np.load(docs_vectors_npy)

    # * * * * * * OLD METHOD (use d2v model)  * * * * * *
    # print("Loading doc2vec model..")
    # d2v_model = doc2vec.Doc2Vec.load(DOC2VEC_MODEL)


    if not isinstance(class_list_doc2vec, list):
        class_list_doc2vec = cfg_emb.load_class_list(class_list_doc2vec)

    for index, vec  in enumerate(output_doc_vectors):
        nv = np.asarray(vec)

        # * * * * * * OLD METHOD (use d2v model)  * * * * * *
        # similars_2 = d2v_model.docvecs.most_similar(positive=[nv], topn=10)
        # similars_2 = np.asarray(similars_2, dtype=np.uint32)
        # # Translate class index of doc2vec (executed on a subset of dataset) in class index of original dataset
        # if class_list_doc2vec is not None:
        #     similars_2 = [int(class_list_doc2vec[s]) for s in similars_2[:, 0]]
        # else:
        #     similars_2 = similars[:, 0]
        #

        # * * * * * * NEW METHOD (use only stored vectors)  * * * * * *
        similars, dists = distances(nv, docs_vectors_npy, get_first_n=top_similars)
        similars = [int(class_list_doc2vec[s]) for s in similars[:]]

        # similars = similars_2  # activate the use of the old method (you need also tu uncomment the d2v_model loading)

        fname = visual_features.fnames[index]
        label = visual_features.labels[index]
        label_name = visual_features.labelIntToStr(label)


        # # Print the images
        # sub_d = imdataset.sub_dataset_from_filename(fname)
        # image = sub_d.data[0]
        # image = image.transpose((2, 0, 1))
        # image = image.transpose((2, 0, 1))
        #plt.title("Class: {} - {}".format(label, label_name) )
        #plt.imshow(image)

        print("")
        print("Class: {} - {}".format(label, str(label_name).decode('utf-8')))
        print("Image: " + str(fname).decode('utf-8'))
        print("Top {} similars classes: ".format(top_similars) + str(similars[:]))
        for i in range(0, top_similars):
            print("{} similar class: {} - {} ".format(i+1, str(similars[i]), visual_features.labelIntToStr(similars[i])))






def test_embedding_map(visual_features, docs_vectors_npy, class_list_doc2vec, im2doc_model,
                       im2doc_model_ext=None, im2doc_weights_ext=None, load_precomputed_imdocs=None,
                       verbose=False, progressbar=True):
    def printv(str):
        if verbose:
            print(str)

    if im2doc_model_ext is None:
        im2doc_model_ext = DEFAULT_IM2DOC_MODEL_EXT
    if load_precomputed_imdocs is None:
        load_precomputed_imdocs = False

    printv("Loading visual features..")
    if not isinstance(visual_features, ImageDataset):
        visual_features = ImageDataset().load_hdf5(visual_features)

    printv("Loading im2doc model..")
    if not isinstance(im2doc_model, Model):
        im2doc_model_name = im2doc_model
        model_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER,
                                  os.path.join(im2doc_model_name, im2doc_model_name + im2doc_model_ext))
        im2doc_model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
    else:
        im2doc_model_name = None

    if im2doc_weights_ext is not None:
        printv("Loading im2doc weights..")
        weight_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER,
                                   os.path.join(im2doc_model, im2doc_model + im2doc_weights_ext))
        im2doc_model.load_weights(weight_file)



    if im2doc_model_name is not None:
        imdocs_path = os.path.join(cfg_emb.IM2DOC_PREDICTION_FOLDER, im2doc_prediction_fname(im2doc_model_name))
    else:
        imdocs_path = "precomputed_imdocs.temp"

    if load_precomputed_imdocs and os.path.exists(imdocs_path):
        printv("Pre computed docs from images found (im2doc embedding)... loading...")
        output_doc_vectors = np.load(imdocs_path)
    else:
        printv("Predict docs from images (im2doc embedding)..")
        im_data = visual_features.data
        while len(im_data.shape) > 2:
            if im_data.shape[-1] == 1:
                im_data = np.squeeze(im_data, axis=(-1,))
        output_doc_vectors = im2doc_model.predict(im_data, verbose=verbose)
        np.save(imdocs_path, output_doc_vectors)

    printv("Loading doc2vec vectors...")
    if not isinstance(docs_vectors_npy, np.ndarray):
        docs_vectors_npy = np.load(docs_vectors_npy)

    if not isinstance(class_list_doc2vec, list):
        class_list_doc2vec = cfg_emb.load_class_list(class_list_doc2vec)

    # mAP test (optimized with cdist)
    if progressbar:
        import sys
        bar = pyprind.ProgBar(len(docs_vectors_npy), stream = sys.stdout)
    av_prec = []
    from scipy.spatial.distance import cdist
    C = cdist(docs_vectors_npy, output_doc_vectors, 'cosine')
    C = 1-C

    for i, dv in enumerate(docs_vectors_npy):
        scores = []
        targets = []
        if progressbar:
            bar.update()

        lbl = int(class_list_doc2vec[i])
        for j, im_label in enumerate(visual_features.labels):
            target = not bool(im_label - lbl)
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



# # mAP test, old method:
# if progressbar:
#     import sys
#     bar = pyprind.ProgBar(len(docs_vectors_npy), stream = sys.stdout)
# av_prec = []
#
# for i, doc in enumerate(docs_vectors_npy):
#
#     scores = []
#     targets = []
#     # scores = np.zeros(len(docs_vectors_npy) * len(visual_features.labels))
#     # targets = np.zeros(len(docs_vectors_npy) * len(visual_features.labels))
#
#     lbl = int(class_list_doc2vec[i])
#     if progressbar:
#         bar.update()
#         #progress(i, len(docs_vectors_npy), "Computing mAP on class {}".format(lbl))
#     j=0
#     for im_docvec, im_label in zip(output_doc_vectors, visual_features.labels):
#         target = not bool(im_label - lbl)
#         score = 1-scipy.spatial.distance.cosine(im_docvec, doc)
#         #score = np.dot(matutils.unitvec(im_docvec), matutils.unitvec(doc))
#         scores.append(score)
#         targets.append(target)
#         # scores[i * len(visual_features.labels) + j] = score
#         # targets[i * len(visual_features.labels) + j] = target
#
#     from sklearn.metrics import average_precision_score
#     AP = average_precision_score(targets, scores)
#     av_prec.append(AP)
#     printv("Class {} - AP = {}".format(lbl, AP))
#
# mAP = np.mean(np.asarray(av_prec))
# printv("\t\tmAP = {}".format(mAP))
# return mAP




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
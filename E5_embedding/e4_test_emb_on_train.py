import os
import sys

import scipy
from gensim import matutils
from gensim import similarities
from gensim.models import doc2vec
import numpy as np
from keras.layers import Dense, K
from keras.models import Model, Sequential, load_model

from E5_embedding import cfg_emb
from config import cfg, common
from imdataset import ImageDataset
import numpy as np




IM_DATASET = 'dbp3120_train_ds'

IM2DOC_MODEL = 'im2docvec_opt-Adadelta_lr-10_bs-32_hl-2000_hl-1000'
IM2DOC_MODEL_EXT = '.model.best.h5'
IM2DOC_WEIGHTS_EXT = None

IM2DOC_PREDICTION_FNAME = 'docs_from_dataset-' + IM_DATASET + '_' + IM2DOC_MODEL + '.npy'



#DOC2VEC_MODEL = "doc2vec_model_train_on_500.bin" # It's not used to compute mAP, only for the verbose test (most similars)
DOC2VEC_MODEL = "doc2vec_model_train_on_400.bin" # It's not used to compute mAP, only for the verbose test (most similars)
CLASS_LIST_D2V_FOR_SIMILARS=cfg_emb.CLASS_LIST_TRAIN
CLASS_LIST_D2V_FOR_mAP = cfg_emb.CLASS_LIST
CLASS_LIST_D2V_FOR_AP=cfg_emb.CLASS_LIST_TRAIN
docs_file = "docvec_400_train_on_400.npy"



def distances(src_vector, cmp_vectors, get_first_n=None):
    dists = []
    args = []
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


def main(args):
    test_embedding_similars()


def test_embedding_similars():

    class_list_doc2vec = cfg_emb.load_class_list(CLASS_LIST_D2V_FOR_SIMILARS)
    print("Loading visual features..")
    visual_features_valid = ImageDataset().load_hdf5(cfg_emb.VISUAL_FEATURES_VALID)
    visual_features = visual_features_valid

    print("Loading doc2vec model..")
    d2v_model = doc2vec.Doc2Vec.load(DOC2VEC_MODEL)

    print("Loading im2doc model..")
    # model = get_model(8000, 300, [4000, 2000])
    # model.load_weights(os.path.join(IM2DOC_MODELS_FOLDER, IM2DOC_MODEL_FNAME))
    model_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(IM2DOC_MODEL, IM2DOC_MODEL + IM2DOC_MODEL_EXT))
    model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
    if IM2DOC_WEIGHTS_EXT is not None:
        print("Loading im2doc weights..")
        weight_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(IM2DOC_MODEL, IM2DOC_MODEL + IM2DOC_WEIGHTS_EXT))
        model.load_weights(weight_file)


    print("Predict docs from images (im2doc embedding)..")
    data = visual_features.data
    while len(data.shape) > 2:
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=(-1,))
    output_doc_vectors = model.predict(data, verbose=True)
    #output_docs_shit = np.load("doc2vec_dbpedia_vectors.npy")

    if not os.path.isdir(cfg_emb.IM2DOC_PREDICTION_FOLDER):
        os.mkdir(cfg_emb.IM2DOC_PREDICTION_FOLDER)
    np.save(os.path.join(cfg_emb.IM2DOC_PREDICTION_FOLDER, IM2DOC_PREDICTION_FNAME), output_doc_vectors)

    # plt.close('all')
    # plt.figure(1)
    # plt.clf()

    for index, vec  in enumerate(output_doc_vectors):
        nv = np.asarray(vec)
        #similars = d2v_model.similar_by_vector(nv)
        similars = d2v_model.docvecs.most_similar(positive=[nv], topn=10)
        similars = np.asarray(similars, dtype=np.uint32)

        # Translate class index of doc2vec (executed on a subset of dataset) in class index of original dataset
        if class_list_doc2vec is not None:
            similars = [int(class_list_doc2vec[s]) for s in similars[:,0]]
        else:
            similars = similars[:,0]

        fname = visual_features.fnames[index]
        label = visual_features.labels[index]
        label_name = visual_features.labelIntToStr(label)

        # sub_d = imdataset.sub_dataset_from_filename(fname)
        # image = sub_d.data[0]
        # image = image.transpose((2, 0, 1))
        # image = image.transpose((2, 0, 1))

        #plt.title("Class: {} - {}".format(label, label_name) )
        #plt.imshow(image)

        print("")
        print("Class: {} - {}".format(label, str(label_name).decode('utf-8')))
        print("Image: " + str(fname).decode('utf-8'))
        print("Top 10 similars classes: " + str(similars[:]))
        for i in range(0, 8):
            print("{} similar class: {} - {} ".format(i+1, str(similars[i]), visual_features.labelIntToStr(similars[i])))






def test_embedding_top_similar(visual_features=cfg_emb.VISUAL_FEATURES_VALID,
                               docs_vectors_npy=docs_file,
                               class_list_doc2vec=CLASS_LIST_D2V_FOR_SIMILARS,
                               im2doc_model=IM2DOC_MODEL,

                               im2doc_model_ext=IM2DOC_MODEL_EXT, im2doc_weights_ext=None,
                               load_precomputed_imdocs=False,
                               top_similars=10):

    print("Loading visual features..")
    if not isinstance(visual_features, ImageDataset):
        visual_features = ImageDataset().load_hdf5(visual_features)


    print("Loading im2doc model..")
    if not isinstance(im2doc_model, Model):
        model_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(im2doc_model, im2doc_model + im2doc_model_ext))
        im2doc_model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
        if im2doc_weights_ext is not None:
            print("Loading im2doc weights..")
            weight_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(im2doc_model, im2doc_model + im2doc_weights_ext))
            im2doc_model.load_weights(weight_file)


    imdocs_path = os.path.join(cfg_emb.IM2DOC_PREDICTION_FOLDER, IM2DOC_PREDICTION_FNAME)
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




def test_embedding_map(visual_features=cfg_emb.VISUAL_FEATURES_VALID,
                       docs_vectors_npy=docs_file,
                       class_list_doc2vec=CLASS_LIST_D2V_FOR_AP,
                       im2doc_model=IM2DOC_MODEL,

                       im2doc_model_ext=IM2DOC_MODEL_EXT, im2doc_weights_ext=None,
                       load_precomputed_imdocs=True):

    print("Loading visual features..")
    if not isinstance(visual_features, ImageDataset):
        visual_features = ImageDataset().load_hdf5(visual_features)


    print("Loading im2doc model..")
    if not isinstance(im2doc_model, Model):
        model_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(im2doc_model, im2doc_model + im2doc_model_ext))
        im2doc_model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
        if im2doc_weights_ext is not None:
            print("Loading im2doc weights..")
            weight_file = os.path.join(cfg_emb.IM2DOC_MODEL_FOLDER, os.path.join(im2doc_model, im2doc_model + im2doc_weights_ext))
            im2doc_model.load_weights(weight_file)


    imdocs_path = os.path.join(cfg_emb.IM2DOC_PREDICTION_FOLDER, IM2DOC_PREDICTION_FNAME)
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

    # TODO: capire perche' qua devo camnbiare la lista (nel caso di doc2vec addestrato sui 500)
    if not isinstance(class_list_doc2vec, list(basestring)):
        class_list_doc2vec = cfg_emb.load_class_list(class_list_doc2vec)


    # mAP test:

    av_prec = []
    for i, doc in enumerate(docs_vectors_npy):
        scores = []
        targets = []
        lbl = int(class_list_doc2vec[i])
        for im_docvec, im_label in zip(output_doc_vectors, visual_features.labels):
            if im_label == lbl:
                target = 1
            else:
                target = 0
            score = np.dot(matutils.unitvec(im_docvec), matutils.unitvec(doc))
            scores.append(score)
            targets.append(target)

        from sklearn.metrics import average_precision_score
        AP = average_precision_score(targets, scores)
        av_prec.append(AP)
        print("Class {} - AP = {}".format(lbl, AP))
    mAP = np.mean(np.asarray(av_prec))
    print("\t\tmAP = {}".format(mAP))








if __name__ == "__main__":
    main(sys.argv)

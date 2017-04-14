import os
import sys

from gensim import matutils
from gensim.models import doc2vec
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.engine import Input
from keras.layers import Dense, K
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adadelta
from matplotlib.pyplot import imshow

from E5_embedding import cfg_emb
from config import cfg, common
from imdataset import ImageDataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



IM_DATASET = 'dbp3120_train_ds'

IM2DOC_MODELS_FOLDER = cfg_emb.IM2DOCVEC_FOLDER
#IM2DOC_MODEL_FNAME = 'video2doc_model_opt-Adadelta_lr-10_bs-32_hl-1000.weights.59.loss-0.0032.h5'
IM2DOC_MODEL = 'im2docvec_opt-Adadelta_lr-10_bs-32_hl-2000_hl-1000'
IM2DOC_MODEL_EXT = '.model.best.h5'
IM2DOC_WEIGHTS_EXT = None


IM2DOC_PREDICTION_FOLDER = 'im2doc_prediction'
IM2DOC_PREDICTION_FNAME = 'docs_from_dataset-' + IM_DATASET + '_' + IM2DOC_MODEL + '.npy'



def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def main(args):
    test_embedding_zero_shot()

# DOC2VEC_MODEL = "doc2vec_model_train_on_400.bin" # It's not used to compute mAP, only for the verbose test (most similars)
# CLASS_LIST_D2V=cfg_emb.CLASS_LIST_TRAIN
# CLASS_LIST_D2V_FOR_AP=cfg_emb.CLASS_LIST_TEST

DOC2VEC_MODEL = "doc2vec_model_train_on_500.bin" # It's not used to compute mAP, only for the verbose test (most similars)
CLASS_LIST_D2V=cfg_emb.CLASS_LIST
CLASS_LIST_D2V_FOR_AP=cfg_emb.CLASS_LIST_TEST


# Extract the test docs from all_docs_file (500) removing the docs presents in the training (400)
# test_docs = all_docs - train_docs
class_list_doc2vec_test = np.asarray(cfg_emb.load_class_list(cfg_emb.CLASS_LIST_TEST), dtype=np.int)
class_list_doc2vec_all = np.asarray(cfg_emb.load_class_list(cfg_emb.CLASS_LIST), dtype=np.int)
index_of_tests_in_all_array = []
for i, cls in enumerate(class_list_doc2vec_all):
    if cls in class_list_doc2vec_test:
        index_of_tests_in_all_array.append(i)
all_docs_file = "docvec_500_train_on_500.npy"
all_docs_array = np.load(all_docs_file)
test_docs_array = all_docs_array[index_of_tests_in_all_array]




def test_embedding_zero_shot():
    import numpy as np

    class_list_doc2vec = cfg_emb.load_class_list(CLASS_LIST_D2V)



    print("Loading visual features..")
    #visual_features = ImageDataset().load_hdf5(cfg_emb.VISUAL_FEATURES_TRAIN)
    #visual_features_valid = ImageDataset().load_hdf5(cfg_emb.VISUAL_FEATURES_VALID)
    visual_features_test = ImageDataset().load_hdf5(cfg_emb.VISUAL_FEATURES_TEST)
    visual_features = visual_features_test

    print("Loading doc2vec model..")
    d2v_model = doc2vec.Doc2Vec.load(DOC2VEC_MODEL)

    print("Loading im2doc model..")
    # model = get_model(8000, 300, [4000, 2000])
    # model.load_weights(os.path.join(IM2DOC_MODELS_FOLDER, IM2DOC_MODEL_FNAME))
    model_file = os.path.join(IM2DOC_MODELS_FOLDER, os.path.join(IM2DOC_MODEL, IM2DOC_MODEL + IM2DOC_MODEL_EXT))
    model = load_model(model_file, custom_objects={'cos_distance': cos_distance})
    if IM2DOC_WEIGHTS_EXT is not None:
        print("Loading im2doc weights..")
        weight_file = os.path.join(IM2DOC_MODELS_FOLDER, os.path.join(IM2DOC_MODEL, IM2DOC_MODEL + IM2DOC_WEIGHTS_EXT))
        model.load_weights(weight_file)


    print("Predict docs from images (im2doc embedding)..")
    data = visual_features.data
    while len(data.shape) > 2:
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=(-1,))
    output_doc_vectors = model.predict(data, verbose=True)
    #output_docs_shit = np.load("doc2vec_dbpedia_vectors.npy")

    if not os.path.isdir(IM2DOC_PREDICTION_FOLDER):
        os.mkdir(IM2DOC_PREDICTION_FOLDER)
    np.save(os.path.join(IM2DOC_PREDICTION_FOLDER, IM2DOC_PREDICTION_FNAME), output_doc_vectors)

    # plt.close('all')
    # plt.figure(1)
    # plt.clf()

    verb = True
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
        if verb:
            #TODO: questo test eseguito con modello doc2vec addestrato sui 500 documenti scazza l'output.. anche se la mAP e' ottima..
            #TODO: risolvere questo problema
            print("")
            print("Class: {} - {}".format(label, str(label_name).decode('utf-8')))
            print("Image: " + str(fname).decode('utf-8'))
            print("Top 10 similars classes: " + str(similars[:]))
            for i in range(0, 8):
                print("{} similar class: {} - {} ".format(i+1, str(similars[i]), visual_features.labelIntToStr(similars[i])))



    # mAP test:
    # TODO: capire perche' qua devo camnbiare la lista (nel caso di doc2vec addestrato sui 500)
    class_list_doc2vec = cfg_emb.load_class_list(CLASS_LIST_D2V_FOR_AP)
    av_prec = []
    for i, doc in enumerate(test_docs_array):
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

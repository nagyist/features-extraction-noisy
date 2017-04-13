import os
import sys

import sklearn
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

DOC2VEC_MODEL = "doc2vec_model_train_on_500.bin"


IM_DATASET = 'dbp3120_train_ds'

IM2DOC_MODELS_FOLDER = cfg_emb.IM2DOCVEC_FOLDER
#IM2DOC_MODEL_FNAME = 'video2doc_model_opt-Adadelta_lr-10_bs-32_hl-1000.weights.59.loss-0.0032.h5'
IM2DOC_MODEL = 'im2docvec_opt-Adadelta_lr-10_bs-32_hl-2000_hl-1000'
IM2DOC_MODEL_EXT = '.model.best.h5'
IM2DOC_WEIGHTS_EXT = '.weights.05.h5'


IM2DOC_PREDICTION_FOLDER = 'im2doc_prediction'
IM2DOC_PREDICTION_FNAME = 'docs_from_dataset-' + IM_DATASET + '_' + IM2DOC_MODEL + '.npy'


def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def main(args):
    test_zero_shot()

def test_zero_shot(class_list_im2doc_train=cfg_emb.CLASS_LIST_TRAIN, class_list_doc2vec_all=cfg_emb.CLASS_LIST):
    import numpy as np

    class_list_im2doc_train = cfg_emb.load_class_list(class_list_im2doc_train)
    class_list_doc2vec_all = cfg_emb.load_class_list(class_list_doc2vec_all)
    for crop_size in cfg.all_crop_size:
        crop = crop_size['crop']
        size = crop_size['size']


        print("Loading visual features..")
        #imdataset = common.dataset('dbp3120_train_ds', crop=crop, size=size, inram=False)
        visual_features_test = ImageDataset().load_hdf5(cfg_emb.VISUAL_FEATURES_TEST)

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
        data = visual_features_test.data
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

        targets = []
        predicted = []

        for index, vec  in enumerate(output_doc_vectors):
            nv = np.asarray(vec)
            #similars = d2v_model.similar_by_vector(nv)
            similars = d2v_model.docvecs.most_similar(positive=[nv], topn=10)
            similars = np.asarray(similars, dtype=np.uint32)

            # Translate class index of doc2vec (executed on a subset of dataset) in class index of original dataset
            if class_list_doc2vec_all is not None:
                similars = [int(class_list_doc2vec_all[s]) for s in similars[:,0]]
            else:
                similars = similars[:,0]

            fname = visual_features_test.fnames[index]
            label = visual_features_test.labels[index]
            label_name = visual_features_test.labelIntToStr(label)

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
                print("{} similar class: {} - {} ".format(i+1, str(similars[i]), visual_features_test.labelIntToStr(similars[i])))

            predicted.append(similars[0])
            targets.append(int(label[0]))

        # TODO:
        from sklearn.metrics import average_precision_score
        mAP = average_precision_score(targets, predicted)
        print("\n\n\n\nmAP = " + str(mAP))


if __name__ == "__main__":
    main(sys.argv)

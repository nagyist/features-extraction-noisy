import os
import sys
from gensim.models import doc2vec
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.engine import Input
from keras.layers import Dense, K
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adadelta
from matplotlib.pyplot import imshow

from E5_embedding.e1_im2doc import get_model, IM2DOC_FOLDER, VISUAL_FEATURES, CLASS_LIST
from config import cfg, common
from imdataset import ImageDataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DOC2VEC_MODEL = "doc2vec_dbpedia_model.bin"


IM_DATASET = 'dbp3120_train_ds'

IM2DOC_MODELS_FOLDER = IM2DOC_FOLDER
#IM2DOC_MODEL_FNAME = 'video2doc_model_opt-Adadelta_lr-10_bs-32_hl-1000.weights.59.loss-0.0032.h5'
IM2DOC_MODEL_FNAME = 'video2doc_model_opt-Adadelta_lr-10_bs-32_hl-2000_hl-1000.weights.40.loss-0.0012.h5'


IM2DOC_PREDICTION_FOLDER = 'im2doc_prediction'
IM2DOC_PREDICTION_FNAME = 'docs_from_dataset-' + IM_DATASET + '_' + IM2DOC_MODEL_FNAME.split('.weights.')[0] + '.npy'


def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def main(args):
    test_embedding()

def test_embedding():
    import numpy as np


    cfg.init('resnet50')
    for crop_size in cfg.all_crop_size:
        crop = crop_size['crop']
        size = crop_size['size']


        print("Loading visual features..")
        imdataset = common.dataset('dbp3120_train_ds', crop=crop, size=size, inram=False)
        visual_features = ImageDataset().load_hdf5(VISUAL_FEATURES)

        print("Loading doc2vec model..")
        d2v_model = doc2vec.Doc2Vec.load(DOC2VEC_MODEL)

        print("Loading im2doc model..")
        # model = get_model(8000, 300, [4000, 2000])
        # model.load_weights(os.path.join(IM2DOC_MODELS_FOLDER, IM2DOC_MODEL_FNAME))
        model = load_model(os.path.join(IM2DOC_MODELS_FOLDER, IM2DOC_MODEL_FNAME),
                           custom_objects={'cos_distance': cos_distance})

        print("Predict docs from images (im2doc embedding)..")
        data = visual_features.data
        while len(data.shape) > 2:
            if data.shape[-1] == 1:
                data = np.squeeze(data, axis=(-1,))
        output_doc_vectors = model.predict(data, verbose=True)
        output_docs_useless = np.load("doc2vec_dbpedia_vectors.npy")

        if not os.path.isdir(IM2DOC_PREDICTION_FOLDER):
            os.mkdir(IM2DOC_PREDICTION_FOLDER)
        np.save(os.path.join(IM2DOC_PREDICTION_FOLDER, IM2DOC_PREDICTION_FNAME), output_doc_vectors)

        # plt.close('all')
        # plt.figure(1)
        # plt.clf()


        for index, vec  in enumerate(output_doc_vectors):
            nv = np.asarray(vec)
            #similars = d2v_model.similar_by_vector(nv)
            similars = d2v_model.docvecs.most_similar(positive=[nv], topn=10)

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
            similars = np.asarray(similars, dtype=np.uint32)
            print("Top 10 similars classes: " + str(similars[:,0]))
            print("1st similar class: {} - {} ".format(str(similars[0,0]), imdataset.labelIntToStr(similars[0,0])))
            print("2nd similar class: {} - {} ".format(str(similars[1,0]), imdataset.labelIntToStr(similars[1,0])))

if __name__ == "__main__":
    main(sys.argv)

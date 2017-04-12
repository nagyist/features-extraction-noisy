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

from E5_embedding import cfg_emb
from E5_embedding.cfg_emb import IM2DOC_FOLDER
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

def test_embedding(class_list_train=cfg_emb.CLASS_LIST_TRAIN, class_list_total=cfg_emb.CLASS_LIST_TRAIN):
    import numpy as np

    if class_list_train is not None:
        class_list_train = file(class_list_train, 'r').read().split('\n')

    class_list_train = cfg_emb.load_class_list(class_list_train)
    class_list_total = cfg_emb.load_class_list(class_list_total)


    cfg.init('resnet50')
    for crop_size in cfg.all_crop_size:
        crop = crop_size['crop']
        size = crop_size['size']


        print("Loading visual features..")
        imdataset = common.dataset('dbp3120_train_ds', crop=crop, size=size, inram=False)
        visual_features = ImageDataset().load_hdf5(cfg_emb.VISUAL_FEATURES_TRAIN)

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
            similars = np.asarray(similars, dtype=np.uint32)

            # Translate class index of doc2vec (executed on a subset of dataset) in class index of original dataset
            if class_list_total is not None:
                similars = [int(class_list_total[s]) for s in similars[:,0]]
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


if __name__ == "__main__":
    main(sys.argv)

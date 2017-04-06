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

from E5_embedding.e1_im2doc import get_model
from config import cfg, common
from imdataset import ImageDataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


im_dataset = 'dbp3120_train_ds'

im2doc_models_folder = 'im2doc_embedding'
im2doc_model_fname = 'video2doc_model_opt-Adadelta_lr-10_bs-32_hl-4000_hl-2000.weights.39.loss-0.0466.h5'

im2doc_prediction_folder = 'im2doc_prediction'
im2doc_prediction_fname = 'docs_from_dataset-' + im_dataset + '_' +im2doc_model_fname.split('.weights.')[0] + '.npy'



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
        visual_features = ImageDataset().load_hdf5("shallow_extracted_features/shallow_feat_dbp3120_train_ds.h5")

        print("Loading doc2vec model..")
        d2v_model = doc2vec.Doc2Vec.load("doc2vec_model_dbpedia.bin")

        print("Loading im2doc model..")
        model = get_model(8000, 300, [4000, 2000])
        model.load_weights(os.path.join(im2doc_models_folder, im2doc_model_fname))
        #model = load_model("video2doc_model.h5")

        print("Predict docs from images (im2doc embedding)..")
        output_doc_vectors = model.predict(visual_features.data, verbose=True)
        output_docs_merdosi = np.load("doc2vec_dbpedia_vectors.npy")

        np.save( os.path.join(im2doc_prediction_folder, im2doc_prediction_fname), output_doc_vectors )

        plt.close('all')
        plt.figure(1)
        plt.clf()

        for index, vec  in enumerate(output_doc_vectors):
            nv = np.asarray(vec)
            #similars = d2v_model.similar_by_vector(nv)
            similars = d2v_model.docvecs.most_similar(positive=[nv], topn=10)

            image = imdataset.data[index]
            label = imdataset.labels[index][0]
            fname = imdataset.fnames[index]
            label_name = imdataset.labelIntToStr(label)
            image = image.transpose((2, 0, 1))
            image = image.transpose((2, 0, 1))

            plt.title("Class: {} - {}".format(label, label_name) )
            plt.imshow(image)

            print("")
            print("Class: {} - {}".format(label, label_name))
            print("Image: " + unicode(fname))
            similars = np.asarray(similars, dtype=np.uint32)
            print("Top 10 similars classes: " + str(similars[:,0]))
            print("Top similar class: {} - {} ".format(str(similars[0,0]), imdataset.labelIntToStr(similars[0,0])))

if __name__ == "__main__":
    main(sys.argv)

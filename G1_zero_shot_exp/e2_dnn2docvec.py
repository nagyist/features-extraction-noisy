import sys
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.engine import Input
from keras.layers import Dense
from keras.metrics import cosine_proximity
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop

from E3_shallow.Shallow import ShallowNetBuilder, ShallowLoader
from G1_zero_shot_exp.e1_prune_classes import pruned_feat_dataset
from config import cfg, common
from imdataset import ImageDataset


def main(args):
    cfg.init()
    dnn2docvec()



def dnn2docvec_exp():
    feat_net='resnet50'
    dataset = cfg.dataset
    shallow_trainset = dataset + '_train'
    shallow_net = ShallowNetBuilder().A()
    shallow_weight_index = 'best'

    SL = ShallowLoader(shallow_trainset, feat_net)
    shallow_net.load(shallowLoader=SL, weight_index=shallow_weight_index)




def dnn2docvec(dataset=cfg.dataset, feat_net='resnet50', shallow_net=ShallowNetBuilder().A()):
    import numpy as np

    print("Loading visual features..")
    SL = ShallowLoader(shallow_training_set, feat_net)
    shallow_net.load(SL, )
    visual_features = pruned_feat_dataset(dataset, feat_net, )
    visual_features = ImageDataset().load_hdf5("shallow_extracted_features/shallow_feat_dbp3120_train_ds.h5")
    print("Loading textual features..")
    textual_features = np.load("doc2vec_dbpedia_vectors.npy")

    print("Generating dataset..")
    data = []
    labels = []
    for lbl, docv in enumerate(textual_features):
        visual_features_with_label = visual_features.sub_dataset_with_label(lbl)
        for visual_feat in visual_features_with_label.data:
            data.append(visual_feat)
            labels.append(docv)

    data = np.asarray(data)
    labels = np.asarray(labels)

    print("Generating model..")

    EPOCHS = 40
    hiddens = [ [1000], [2000], [4000], [2000,1000], [4000, 2000], [4000, 2000, 1000]]
    lrs = [10, 1]
    batch_sizes = [64, 32, 16]
    optimizers_str = ['Adadelta', 'Adagrad']
    optimizers = [Adadelta, Adagrad]
    for hid in hiddens:
        for opt, opt_str in zip(optimizers, optimizers_str):
            for lr in lrs:
                for bs in batch_sizes:

                    print ""
                    print("Training model..")
                    print "hiddens: " + str(hid)
                    print "optim:   " + str(opt_str)
                    print "lr:      " + str(lr)

                    fname = "video2doc_model_opt-{}_lr-{}_bs-{}".format(opt_str, lr, bs)
                    for i, hu in enumerate(hid):
                        fname += "_hl-" + str(hu)

                    model = get_model(data.shape[-1], labels.shape[-1], hid)
                    model.compile(optimizer=opt(lr=lr), loss=cos_distance)

                    earlystop = EarlyStopping(monitor='loss', min_delta=0.0005, patience=8)
                    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, epsilon=0.001)
                    bestpoint = ModelCheckpoint(fname + '.weights.{epoch:02d}.loss-{loss:.4f}.h5', monitor='loss',
                                                save_best_only=True, save_weights_only=False)
                    bestpoint_wo = ModelCheckpoint(fname + '.weights.{epoch:02d}.loss-{loss:.4f}.h5', monitor='loss',
                                                save_best_only=True, save_weights_only=True)
                    callbacks = [earlystop, reduceLR, bestpoint, bestpoint_wo]
                    model.fit(data, labels, batch_size=64, nb_epoch=EPOCHS, verbose=1, shuffle=True, callbacks=callbacks)




if __name__ == "__main__":
    main(sys.argv)

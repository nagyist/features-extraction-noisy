import keras
from test_embedding import test_embedding_map


class ModelMAP(keras.callbacks.Callback):
    def __init__(self, visual_features, docs_vectors, class_list,
                 im2doc_model_ext=None, im2doc_weights_ext=None, load_precomputed_imdocs=None,
                 history_key="mAP"):
        super(ModelMAP, self).__init__()
        self.visual_features = visual_features
        self.docs_vectors = docs_vectors
        self.class_list = class_list
        self.im2doc_model_ext = im2doc_model_ext
        self.im2doc_weights_ext =im2doc_weights_ext
        self.load_precomputed_imdocs = load_precomputed_imdocs
        self.history_key=history_key

    def on_train_begin(self, logs={}):
        self.avarage_precisions = []
        self.mAP = None

    def on_epoch_end(self, epoch, logs={}):
        print("\nComputing " + self.history_key)
        mAP = test_embedding_map(visual_features=self.visual_features,
                                 class_list_doc2vec=self.class_list,
                                 docs_vectors_npy=self.docs_vectors,
                                 im2doc_model=self.model,
                                 im2doc_model_ext=self.im2doc_weights_ext,
                                 im2doc_weights_ext=self.im2doc_weights_ext,
                                 load_precomputed_imdocs=self.load_precomputed_imdocs,
                                 verbose=False)
        logs[self.history_key] = mAP
        print("    {}: {}\n\n".format(self.history_key, mAP))





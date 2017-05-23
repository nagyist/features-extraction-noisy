import keras
from test_embedding import test_embedding_tx_mAP, test_embedding_im_mAP

from E6_joint_model.test_joint import retrieve_image_map


class ModelMAP(keras.callbacks.Callback):
    def __init__(self, visual_features, docs_vectors, class_list,
                 im2doc_model_ext=None, im2doc_weights_ext=None, load_precomputed_imdocs=None,
                 history_key="mAP",
                 exe_on_train_begin=False, on_train_begin_key=None,
                 exe_on_batch_end=False, on_batch_end_key=None,
                 im_map=True, tx_map=True):
        super(ModelMAP, self).__init__()
        self._visual_features = visual_features
        self._docs_vectors = docs_vectors
        self._class_list = class_list
        self._im2doc_model_ext = im2doc_model_ext
        self._im2doc_weights_ext =im2doc_weights_ext
        self._load_precomputed_imdocs = load_precomputed_imdocs
        self._history_key=history_key
        self._exe_on_train_begin = exe_on_train_begin
        self._exe_on_batch_end = exe_on_batch_end

        if on_batch_end_key is None:
            self.on_batch_end_key = 'batch_end-' + history_key
        else:
            self.on_batch_end_key = on_batch_end_key

        if on_train_begin_key is None:
            self.on_train_begin_key = 'train_begin-' + history_key
        else:
            self.on_train_begin_key = on_train_begin_key

        self.im_map=im_map
        self.tx_map=tx_map

    def on_train_begin(self, logs={}):
        self.batch_mAP_hist = []
        self.train_begin_mAP = None
        if self._exe_on_train_begin:
            mAP = self._compute_map(self.on_train_begin_key)
            logs[self.on_train_begin_key] = mAP
            self.train_begin_mAP = mAP
        return self.train_begin_mAP

    def on_epoch_end(self, epoch, logs={}):
        mAP = self._compute_map(key=self._history_key)
        logs[self.on_epoch_end] = mAP
        return mAP


    def on_batch_end(self, batch, logs={}):
        if self._exe_on_batch_end:
            mAP = self._compute_map(key=self.on_batch_end_key)
            self.batch_mAP_hist.append(mAP)
            return mAP
        else:
            return None

    def _compute_map(self, key):
        print("\nComputing " + key)
        if self.im_map:
            mAP = test_embedding_tx_mAP(visual_features=self._visual_features,
                                        class_list_doc2vec=self._class_list,
                                        docs_vectors_npy=self._docs_vectors,
                                        im2doc_model=self.model,
                                        im2doc_model_ext=self._im2doc_weights_ext,
                                        im2doc_weights_ext=self._im2doc_weights_ext,
                                        load_precomputed_imdocs=self._load_precomputed_imdocs,
                                        verbose=False)
            print("    TX MAP, {}: {}\n\n".format(key, mAP))

        if self.tx_map:
            mAP = test_embedding_im_mAP(visual_features=self._visual_features,
                                        class_list_doc2vec=self._class_list,
                                        docs_vectors_npy=self._docs_vectors,
                                        im2doc_model=self.model,
                                        im2doc_model_ext=self._im2doc_weights_ext,
                                        im2doc_weights_ext=self._im2doc_weights_ext,
                                        load_precomputed_imdocs=self._load_precomputed_imdocs,
                                        verbose=False)
            print("    IM MAP, {}: {}\n\n".format(key, mAP))






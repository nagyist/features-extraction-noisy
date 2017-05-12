import keras
from E6_joint_model.test_joint import test_joint_map


class ModelMAP(keras.callbacks.Callback):
    def __init__(self, visual_features, docs_vectors, class_list,
                 model_ext=None, model_weights_ext=None, load_precomputed_imdocs=None,
                 history_key="mAP",
                 exe_on_train_begin=False, on_train_begin_key=None,
                 exe_on_epoch_end=False,
                 exe_on_batch_end=False, on_batch_end_key=None,
                 exe_on_train_end=False):
        super(ModelMAP, self).__init__()
        self._visual_feat_vecs = visual_features
        self._docs_feat_vecs = docs_vectors
        self._class_list = class_list
        self._model_ext = model_ext
        self._model_weights_ext =model_weights_ext
        self._load_precomputed_imdocs = load_precomputed_imdocs
        self._history_key=history_key
        self._exe_on_train_begin = exe_on_train_begin
        self._exe_on_train_end = exe_on_train_end
        self._exe_on_batch_end = exe_on_batch_end
        self._exe_on_epoch_end = exe_on_epoch_end
        if on_batch_end_key is None:
            self.on_batch_end_key = 'batch_end-' + history_key
        else:
            self.on_batch_end_key = on_batch_end_key

        if on_train_begin_key is None:
            self.on_train_begin_key = 'train_begin-' + history_key
        else:
            self.on_train_begin_key = on_train_begin_key


    def on_train_begin(self, logs={}):
        self.batch_mAP_hist = []
        self.train_begin_mAP = None
        if self._exe_on_train_begin:
            mAP = self._compute_map(self.on_train_begin_key)
            logs[self.on_train_begin_key] = mAP
            self.train_begin_mAP = mAP
        return self.train_begin_mAP

    def on_train_end(self, logs={}):
        if self._exe_on_train_end:
            mAP = self._compute_map('mAP-on_train_end')
            self.train_end_mAP = mAP
        return self.train_end_mAP


    def on_epoch_end(self, epoch, logs={}):
        if self._exe_on_epoch_end:
            mAP = self._compute_map(key=self._history_key)
            logs[self.on_epoch_end] = mAP
            return mAP
        else:
            return None

    def on_batch_end(self, batch, logs={}):
        if self._exe_on_batch_end:
            mAP = self._compute_map(key=self.on_batch_end_key)
            self.batch_mAP_hist.append(mAP)
            return mAP
        else:
            return None

    def _compute_map(self, key):
        print("\nComputing " + key)
        mAP = test_joint_map(img_features=self._visual_feat_vecs, txt_features=self._docs_feat_vecs,
                             class_list_doc2vec=self._class_list, joint_model=self.model,
                             joint_model_ext=self._model_ext, joint_model_weights_ext=self._model_weights_ext,
                             #verbose=True, progressbar=False)
                             verbose=False, progressbar=True)

        print("    {}: {}\n\n".format(key, mAP))






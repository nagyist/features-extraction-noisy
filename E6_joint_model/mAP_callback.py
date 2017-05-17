import keras
from E6_joint_model.test_joint import test_joint_map


class ModelMAP(keras.callbacks.Callback):
    def __init__(self, visual_features, docs_vectors, class_list,
                 model_ext=None, model_weights_ext=None, load_precomputed_imdocs=None,
                 history_key="mAP",
                 exe_fit_begin=False,
                 exe_fit_end=False,
                 exe_epoch_begin_period=0,
                 exe_epoch_end_period=0,
                 exe_batch_begin_period=0,
                 exe_batch_end_period=0,
                 mode='text-retrivial'
                 ):
        '''
        
        :param visual_features: 
        :param docs_vectors: 
        :param class_list: 
        :param model_ext: 
        :param model_weights_ext: 
        :param load_precomputed_imdocs: 
        :param history_key: 
        :param exe_fit_begin: 
        :param exe_fit_end: 
        :param exe_epoch_begin_period: 
        :param exe_epoch_end_period: 
        :param exe_batch_begin_period: 
        :param exe_batch_end_period: 
        :param mode: 'text-retrivial' or 'image-retrivial'
        '''
        super(ModelMAP, self).__init__()
        self._visual_feat_vecs = visual_features
        self._docs_feat_vecs = docs_vectors
        self._class_list = class_list
        self._model_ext = model_ext
        self._model_weights_ext =model_weights_ext
        self._load_precomputed_imdocs = load_precomputed_imdocs
        self.key=history_key

        self.batch_end_history = []
        self.batch_beg_history = []
        self.epoch_beg_history = []
        self.epoch_end_history = []
        self.fit_beg_history = None
        self.fit_end_history = None

        self._exe_fit_beg = exe_fit_begin
        self._exe_fit_end = exe_fit_end
        self._exe_epoch_end = exe_epoch_begin_period
        self._exe_epoch_beg = exe_epoch_end_period
        self._exe_batch_beg = exe_batch_begin_period
        self._exe_batch_end = exe_batch_end_period

        self._batch_counter = 0


    def on_train_begin(self, logs=None):
        if self._exe_fit_beg:
            mAP = self._compute_map(self.key + '-fit-begin')
            self.fit_beg_history = mAP

    def on_train_end(self, logs=None):
        if self._exe_fit_end:
            mAP = self._compute_map(self.key + '-fit-end')
            self.fit_end_history = mAP

    def on_epoch_begin(self, epoch, logs=None):
        if self._exe_epoch_end>0 and self._exe_epoch_end%epoch == 0:
            mAP = self._compute_map(self.key + '-epoch-begin')
            logs[self.key + '-epoch-begin'] = mAP
            self.epoch_beg_history.append(mAP)

    def on_epoch_end(self, epoch, logs=None):
        if self._exe_epoch_end>0 and self._exe_epoch_end%epoch == 0:
            mAP = self._compute_map(self.key + '-epoch-end')
            logs[self.key + '-epoch-end'] = mAP
            self.epoch_end_history.append(mAP)

    def on_batch_begin(self, batch, logs=None):
        self._batch_counter += 1
        if self._exe_batch_end>0 and self._batch_counter%self._exe_epoch_end == 0:
            mAP = self._compute_map(self.key + '-batch-begin')
            self.batch_beg_history.append(mAP)

    def on_batch_end(self, batch, logs=None):
        if self._exe_batch_end>0 and self._batch_counter%self._exe_epoch_end == 0:
            mAP = self._compute_map(self.key + '-batch-end')
            self.batch_end_history.append(mAP)



    def _compute_map(self, key):
        print("\nComputing: " + key)
        mAP = test_joint_map(img_features=self._visual_feat_vecs,
                             txt_features=self._docs_feat_vecs,
                             class_list_doc2vec=self._class_list,
                             joint_model=self.model,
                             joint_model_ext=self._model_ext,
                             joint_model_weights_ext=self._model_weights_ext,
                             #verbose=True, progressbar=False)
                             verbose=False, progressbar=True)
        print("    {}: {}\n\n".format(key, mAP))
        return mAP






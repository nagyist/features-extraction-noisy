import keras
from zzE6_joint_model.test_joint import retrieve_image_map, recall_top_k


class ModelMAP(keras.callbacks.Callback):
    def __init__(self, visual_features, docs_vectors, class_list,
                 model_ext=None, model_weights_ext=None, load_precomputed_imdocs=None,
                 data_name="dataset",
                 exe_fit_begin=False,
                 exe_fit_end=False,
                 exe_epoch_begin_period=0,
                 exe_epoch_end_period=0,
                 exe_batch_begin_period=0,
                 exe_batch_end_period=0,
                 text_retrieval_map=True,
                 image_retrieval_map=False,
                 recall_at_k=[]
                 ):
        '''
        
        :param visual_features: 
        :param docs_vectors: 
        :param class_list: 
        :param model_ext: 
        :param model_weights_ext: 
        :param load_precomputed_imdocs: 
        :param data_name: 
        :param exe_fit_begin: 
        :param exe_fit_end: 
        :param exe_epoch_begin_period: 
        :param exe_epoch_end_period: 
        :param exe_batch_begin_period: 
        :param exe_batch_end_period: 
        :param mode:
        '''
        super(ModelMAP, self).__init__()
        self._visual_feat_vecs = visual_features
        self._docs_feat_vecs = docs_vectors
        self._class_list = class_list
        self._model_ext = model_ext
        self._model_weights_ext =model_weights_ext
        self._load_precomputed_imdocs = load_precomputed_imdocs
        self.data_name=data_name

        self.batch_end_history = []
        self.batch_beg_history = []
        self.epoch_beg_history = []
        self.epoch_end_history = []
        self.fit_beg_history = {}
        self.fit_end_history = {}

        self._exe_fit_beg = exe_fit_begin
        self._exe_fit_end = exe_fit_end
        self._exe_epoch_end = exe_epoch_begin_period
        self._exe_epoch_beg = exe_epoch_end_period
        self._exe_batch_beg = exe_batch_begin_period
        self._exe_batch_end = exe_batch_end_period

        if isinstance(recall_at_k, int):
            self.recall_at_k = [recall_at_k]
        else:
            self.recall_at_k = recall_at_k
        self.text_retrieval_map = text_retrieval_map
        self.image_retrieval_map = image_retrieval_map
        self._batch_counter = 0


    def on_train_begin(self, logs=None):
        if self._exe_fit_beg:
            hist_step = self._compute_map(self.data_name + ' at fit-begin')
            self.fit_beg_history = hist_step

    def on_train_end(self, logs=None):
        if self._exe_fit_end:
            hist_step = self._compute_map(self.data_name + ' at fit-end')
            self.fit_end_history = hist_step

    def on_epoch_begin(self, epoch, logs=None):
        if self._exe_epoch_end>0 and self._exe_epoch_end%epoch == 0:
            hist_step = self._compute_map(self.data_name + ' at epoch-begin')
           # logs[self.data_name + '-epoch-begin'] = hist_step
            self.epoch_beg_history.append(hist_step)

    def on_epoch_end(self, epoch, logs=None):
        if self._exe_epoch_end>0 and self._exe_epoch_end%epoch == 0:
            hist_step = self._compute_map(self.data_name + ' at epoch-end')
           # logs[self.data_name + '-epoch-end'] = hist_step
            self.epoch_end_history.append(hist_step)

    def on_batch_begin(self, batch, logs=None):
        self._batch_counter += 1
        if self._exe_batch_end>0 and self._batch_counter%self._exe_epoch_end == 0:
            hist_step = self._compute_map(self.data_name + ' at batch-begin')
            self.batch_beg_history.append(hist_step)

    def on_batch_end(self, batch, logs=None):
        if self._exe_batch_end>0 and self._batch_counter%self._exe_epoch_end == 0:
            hist_step = self._compute_map(self.data_name + ' at batch-end')
            self.batch_end_history.append(hist_step)



    def _compute_map(self, data_and_moment):

        hist_step = {}
        if self.image_retrieval_map:
            print("\nComputing image retrieval mAP on {}...".format(data_and_moment))
            mAP = retrieve_image_map(img_features=self._visual_feat_vecs,
                                    txt_features=self._docs_feat_vecs,
                                    class_list_doc2vec=self._class_list,
                                    joint_model=self.model,
                                    joint_model_ext=self._model_ext,
                                    joint_model_weights_ext=self._model_weights_ext,
                                    #verbose=True, progressbar=False)
                                    verbose=False, progressbar=True)
            hist_step['map-im-retrieval'] = mAP
            print("Image retrieval mAP on {}: {}".format(data_and_moment, mAP))

        if self.text_retrieval_map:
            print("\nComputing text retrieval mAP on {}...".format(data_and_moment))
            mAP = retrieve_image_map(img_features=self._visual_feat_vecs,
                                    txt_features=self._docs_feat_vecs,
                                    class_list_doc2vec=self._class_list,
                                    joint_model=self.model,
                                    joint_model_ext=self._model_ext,
                                    joint_model_weights_ext=self._model_weights_ext,
                                    #verbose=True, progressbar=False)
                                    verbose=False, progressbar=True)
            hist_step['map-tx-retrieval'] = mAP
            print("Text retrieval mAP on {}: {}".format(data_and_moment, mAP))

        for k in self.recall_at_k:

            print("\nComputing text recall@{} on {}...".format(k, data_and_moment))
            rec = recall_top_k(img_features=self._visual_feat_vecs,
                               txt_features=self._docs_feat_vecs,
                               class_list_doc2vec=self._class_list,
                               joint_model=self.model,
                               joint_model_ext=self._model_ext,
                               joint_model_weights_ext=self._model_weights_ext,
                               #verbose=True, progressbar=False)
                               verbose=False, progressbar=True,
                               top_k=k)
            hist_step['avg-recall@{}'.format(k)] = rec
            print("Text recall@{} on {}: {}".format(k, data_and_moment, rec))

        return hist_step






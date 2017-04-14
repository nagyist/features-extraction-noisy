import keras
import numpy as np

# TODO !!!
class LossHistory(keras.callbacks.Callback):
    def __init__(self, test_doc_vectors):
        self.test_doc_vectors = test_doc_vectors

    def on_train_begin(self, logs={}):
        self.avarage_precisions = []
        self.mAP = None

    def on_batch_end(self, batch, logs={}):
        self.model
        for index, vec in enumerate(self.test_doc_vectors):
            nv = np.asarray(vec)
            # similars = d2v_model.similar_by_vector(nv)
            similars = d2v_model.docvecs.most_similar(positive=[nv], topn=10)
            similars = np.asarray(similars, dtype=np.uint32)

            # Translate class index of doc2vec (executed on a subset of dataset) in class index of original dataset
            if class_list_doc2vec is not None:
                similars = [int(class_list_doc2vec[s]) for s in similars[:, 0]]
            else:
                similars = similars[:, 0]

            fname = visual_features.fnames[index]
            label = visual_features.labels[index]
            label_name = visual_features.labelIntToStr(label)

        # mAP test:
        # TODO: capire perche' qua devo camnbiare la lista (nel caso di doc2vec addestrato sui 500)
        class_list_doc2vec = cfg_emb.load_class_list(CLASS_LIST_D2V_FOR_AP)
        av_prec = []
        for i, doc in enumerate(test_docs_array):
            scores = []
            targets = []
            lbl = int(class_list_doc2vec[i])
            for im_docvec, im_label in zip(self.test_doc_vectors, visual_features.labels):
                if im_label == lbl:
                    target = 1
                else:
                    target = 0
                score = np.dot(matutils.unitvec(im_docvec), matutils.unitvec(doc))
                scores.append(score)
                targets.append(target)

            from sklearn.metrics import average_precision_score
            AP = average_precision_score(targets, scores)
            av_prec.append(AP)
            print("Class {} - AP = {}".format(lbl, AP))
        mAP = np.mean(np.asarray(av_prec))
        print("\t\tmAP = {}".format(mAP))
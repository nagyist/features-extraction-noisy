import sys
from E5_embedding import cfg_emb
import numpy as np

from test_embedding import test_embedding_map, test_embedding_top_similars



IM2DOC_MODEL = 'im2docvec_opt-Adadelta_lr-10_bs-32_hl-2000_hl-1000'

DOCS_FILE_400 = "docvec_400_train_on_400.npy"
DOCS_FILE_500 = "docvec_500_train_on_500.npy"


VISUAL_FEATURES_TRAIN = cfg_emb.VISUAL_FEATURES_TRAIN
VISUAL_FEATURES_VALID = cfg_emb.VISUAL_FEATURES_VALID
CLASS_LIST_D2V_for_similars=cfg_emb.CLASS_LIST_400
CLASS_LIST_D2V_for_map=cfg_emb.CLASS_LIST_400

ZERO_SHOT_VISUAL_FEATURES = cfg_emb.VISUAL_FEATURES_TEST
ZERO_SHOT_CLASS_LIST_for_map = cfg_emb.CLASS_LIST_100
ZERO_SHOT_CLASS_LIST_all = cfg_emb.CLASS_LIST_500




def main(args):
    test_embedding_on_valid()





def test_embedding_on_train(map=True, top_similar_output=None):
    if isinstance(top_similar_output, int):
        test_embedding_top_similars(visual_features=cfg_emb.VISUAL_FEATURES_TRAIN,
                                    class_list_doc2vec=cfg_emb.CLASS_LIST_400,
                                    docs_vectors_npy=DOCS_FILE_400,
                                    im2doc_model_name=IM2DOC_MODEL,
                                    im2doc_model_ext=None, im2doc_weights_ext=None,
                                    load_precomputed_imdocs=False,
                                    top_similars=top_similar_output)
    if map:
        test_embedding_map(visual_features=cfg_emb.VISUAL_FEATURES_TRAIN,
                           class_list_doc2vec=cfg_emb.CLASS_LIST_400,
                           docs_vectors_npy=DOCS_FILE_400,
                           im2doc_model=IM2DOC_MODEL,
                           im2doc_model_ext=None, im2doc_weights_ext=None,
                           load_precomputed_imdocs=True)



def test_embedding_on_valid(map=True, top_similar_output=None):
    if isinstance(top_similar_output, int):
        test_embedding_top_similars(visual_features=VISUAL_FEATURES_VALID,
                                    docs_vectors_npy=DOCS_FILE_400,
                                    class_list_doc2vec=CLASS_LIST_D2V_for_similars,
                                    im2doc_model_name=IM2DOC_MODEL,
                                    im2doc_model_ext=None, im2doc_weights_ext=None,
                                    load_precomputed_imdocs=False,
                                    top_similars=top_similar_output)
    if map:
        test_embedding_map(visual_features=VISUAL_FEATURES_VALID,
                           docs_vectors_npy=DOCS_FILE_400,
                           class_list_doc2vec=CLASS_LIST_D2V_for_map,
                           im2doc_model=IM2DOC_MODEL,
                           im2doc_model_ext=None, im2doc_weights_ext=None,
                           load_precomputed_imdocs=True)




def test_embedding_zero_shot(map=True, top_similar_output=None):
    docs_vectors_500 = np.load(DOCS_FILE_500)
    docs_vectors_100_zero_shot = []
    class_list_all = cfg_emb.load_class_list(ZERO_SHOT_CLASS_LIST_all)
    class_list_for_map = cfg_emb.load_class_list(ZERO_SHOT_CLASS_LIST_for_map)
    for i, cls in enumerate(class_list_all):
        if cls in class_list_for_map:
            docs_vectors_100_zero_shot.append(docs_vectors_500[i])
    docs_vectors_100_zero_shot = np.asarray(docs_vectors_100_zero_shot)

    if isinstance(top_similar_output, int):
        print("Top similars on zero shot aproach: not yet implemented.")
        # test_embedding_top_similars(visual_features=cfg_emb.VISUAL_FEATURES_VALID,
        #                             docs_vectors_npy=docs_vectors_100_zero_shot,
        #                             class_list_doc2vec=CLASS_LIST_D2V_for_similars,
        #                             im2doc_model=IM2DOC_MODEL,
        #                             im2doc_model_ext=IM2DOC_MODEL_EXT, im2doc_weights_ext=None,
        #                             load_precomputed_imdocs=False,
        #                             top_similars=top_similar_output)
    if map:
        test_embedding_map(visual_features=ZERO_SHOT_VISUAL_FEATURES,
                           docs_vectors_npy=docs_vectors_100_zero_shot,
                           class_list_doc2vec=ZERO_SHOT_CLASS_LIST_for_map,
                           im2doc_model=IM2DOC_MODEL,
                           im2doc_model_ext=None, im2doc_weights_ext=None,
                           load_precomputed_imdocs=True)







if __name__ == "__main__":
    main(sys.argv)
import os
os.environ['THEANO_FLAGS'] = "device=gpu1"

from E5_embedding.e2_test_embedding import test_embedding_zero_shot, test_embedding_on_valid, test_embedding_on_train



#test_embedding_zero_shot()
#test_embedding_on_valid()
test_embedding_on_train(im_map=True)
#test_embedding_on_train(map=False, top_similar_output=1)

import os
os.environ['THEANO_FLAGS'] = "floatX=float32,device=gpu1,lib.cnmem=0, exception_verbosity=high"
os.environ
from E5_embedding.e1_im2doc_with_mAP import im2docvec_wvalid_map


#im2doc()
im2docvec_wvalid_map()

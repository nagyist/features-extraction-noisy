import os
# floatX=float32,device=gpu1,lib.cnmem=1,
# os.environ['THEANO_FLAGS'] = "exception_verbosity=high, optimizer=None"
import random

from keras.metrics import cosine_proximity

os.environ['KERAS_BACKEND'] = "tensorflow"

from keras.layers import K
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np



def contrastive_loss_over_distance(labels, distances):
    '''
    :param labels: 1D tensor containing 0 or 1 for each example
    :param distances: 
    :return: 
    '''
    margin=1
    # loss = K.mean((distances + K.maximum(margin-shifted_distances, 0)))
    loss = labels*K.square(distances) + (1-labels) * K.square(K.maximum(margin - distances, 0))
    #loss = K.mean(loss)
    # loss = K.mean(distances - shifted_distances)
    return loss


def contrastive_test():
    #               batch size = 3, space dim = 2
    t1 = K.variable(np.array( [[.1, .9], [.1, .9], [.2, .8], [.2, .8], [.3, .7], [.3, .7]] ))
    t2 = K.variable(np.array( [[.0, .0], [.3, .0], [.3, .5], [.9, .5], [.4, .9], [.7, .6]] ))
    t3 = K.variable(np.array( [[.1, .9], [.3, .0], [.2, .8], [.9, .5], [.3, .7], [.7, .6]] ))
    t4 = K.variable(np.array( [[.0, .0], [.0, .0], [.0, .0], [.0, .0], [.0, .0], [.0, .0]] ))

    contr_label =  np.array(  [  1,        0,        1,        0,        1,        0 ] )
    contr_label = K.variable(contr_label)

    d1 = my_distance([t1, t2])
    d2 = my_distance([t1, t3])
    d3 = my_distance([t1, t4])

    l1 = contrastive_loss_over_distance(contr_label, d1)
    l1 = K.eval(l1)
    l2 = contrastive_loss_over_distance(contr_label, d2)
    l2 = K.eval(l2)
    l3 = contrastive_loss_over_distance(contr_label, d3)
    l3 = K.eval(l3)

    print 'loss 1: ' + str(l1)
    print 'loss 2: ' + str(l2)
    print 'loss 3: ' + str(l3)


#contrastive_test()


def my_cosine_proximity(tensors):
    a = K.l2_normalize(tensors[0], axis=-1)
    b = K.l2_normalize(tensors[1], axis=-1)
    return 1-K.mean(a * b)





def cos_distance_test():
    # t1 = K.variable(np.array([[1, 1], [1, 1]]))
    # t2 = K.variable(np.array([[1, 1], [1, 1]]))

    t1 = K.variable(np.array([1, 0]))
    t2 = K.variable(np.array([0, 1]))


    d = 1+cosine_proximity(t1, t2)
    d = K.eval(d)
    print 'Cos distance: ' + str(d)
cos_distance_test()






def my_distance(tensors):
    if (len(tensors) != 2):
        raise 'oops'
    dist = K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1))
    return dist


def distance_test():
    # t1 = K.variable(np.array([[1, 0], [1, 0]]))
    # t2 = K.variable(np.array([[0, 1], [0, 0]]))

    t1 = K.variable(np.array([1, 0]))
    t2 = K.variable(np.array([0, 1]))

    d = my_distance([t1, t2])
    d = K.eval(d)
    print 'distance: ' + str(d)
#distance_test()

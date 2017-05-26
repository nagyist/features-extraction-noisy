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
#cos_distance_test()






def my_distance(tensors):
    if (len(tensors) != 2):
        raise 'oops'
    dist = K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1))
    return dist


def distance_test():
    t1 = K.variable(np.array([[1, 0], [1, 0]]))
    t2 = K.variable(np.array([[0, 1], [0, 0]]))

    #t1 = K.variable(np.array([1, 0]))
    #t2 = K.variable(np.array([0, 1]))

    d = my_distance([t1, t2])
    d = K.eval(d)
    print 'distance: ' + str(d)
#distance_test()




def test_contrastive_loss_over_distances():
    t1 = K.variable(np.array([[0.1, 0.2, 0.9, 0.8], [0.9, 0.95, 0.2, 0.4]]))
    t2 = K.variable(np.array([[0.12, 0.21, 0.89, 0.78], [0.12, 0.21, 0.89, 0.78]]))

    d = my_distance([t1, t2])
    print 'distance: ' + str(K.eval(d))

    labels = np.array([0, 0])
    labels_2 = np.array([1, 1])
    labels_3 = np.array([1, 0])
    labels_4 = np.array([0, 1])

    #loss = contrastive_loss_over_distance(labels, d)
    #loss2 = contrastive_loss_over_distance(labels_2, d)
    loss3 = contrastive_loss_over_distance(labels_3, d)
    loss4 = contrastive_loss_over_distance(labels_4, d)

#    print "loss: " + str(K.eval(loss))
 #   print "loss2: " + str(K.eval(loss2))
    print "loss3: " + str(K.eval(loss3))
    print "loss4: " + str(K.eval(loss4))




def contrastive_loss_over_distance(labels, distances):
    '''
    :param labels: 1D tensor containing 0 or 1 for each example
    :param distances: 
    :return: 
    '''
    margin=1
    # loss = K.mean((distances + K.maximum(margin-shifted_distances, 0)))
    print(K.eval(distances))
    right = margin - distances
    print(K.eval(right))
    right = K.maximum(right, 0)
    print(K.eval(right))
    right = K.square(right)
    print(K.eval(right))

    print ""

    print(K.eval(distances))
    left = distances
    print(K.eval(left))
    left = K.square(left)
    print(K.eval(left))

    left = labels*left
    print(K.eval(left))
    right = (1-labels)*right
    print(K.eval(right))

    loss = K.mean(left + right)
    print(K.eval(loss))


    # loss = K.mean(distances - shifted_distances)
    return loss

test_contrastive_loss_over_distances()
from keras.engine import Input, Model, merge, Merge
from keras.layers import Dense, LSTM, K, Activation
from keras.metrics import mean_squared_error, cosine_proximity
from keras.models import Sequential
import numpy as np


def get_sub_model(joint_model, submodel):
    # type: (Model, basestring) -> Model
    im_input = joint_model.input[0]
    tx_input = joint_model.input[1]
    im_emb = joint_model.output[0]
    tx_emb = joint_model.output[1]

    optimizer = joint_model.optimizer
    if submodel is not None:
        if submodel == "img":
            model = Model(input=im_input, output=im_emb)
            model.compile(optimizer=optimizer, loss=fake_loss)
        elif submodel == "txt":
            model = Model(input=tx_input, output=tx_emb)
            model.compile(optimizer=optimizer, loss=avg_batch_mse_loss)
        else:
            model = None
    return model

class JointEmbedder():

    # Model Idea:
    #
    # |   fc
    # | ------------->  | = t \
    # |  300 -> 128            \
    #                           >- [i|t]
    # |                        /
    # |   fc                  /
    # | ------------->  | = i
    # |  2K -> 128
    # |

    def __init__(self, im_dim, tx_dim, out_dim):
        self.im_input_dim = im_dim
        self.tx_input_dim = tx_dim
        self.output_dim = out_dim


    def model(self, optimizer, activation=None, im_hidden_layers=None, tx_hidden_layers=None,
              im_hidden_activation=None, tx_hidden_activation=None,
              tx_tx_factr=1, tx_im_factr=1, submodel=None):
        '''
        
        :param optimizer: 
        :param hidden_activation: activation for both text and image last dense layers
        :param im_hidden_layers: a list of numbers, each one of them is the number of hidden unit for a new hidden layer
        :param tx_hidden_layers: a list of numbers, each one of them is the number of hidden unit for a new hidden layer
        :param im_hidden_activation: a list of activation functions, one for each image hidden layer, None to disable
        :param tx_hidden_activation: a list of activation functions, one for each text hidden layer, None to disable
        :param tx_tx_factr: weight for the loss text vs text
        :param tx_im_factr: weight for the loss image vs text
        :param submodel: None to get the whole model, with 3 outputs: text, image, and (text-image) embedding output. 
                             'txt' to get only the text leg submodel, with only the text-embedding output. 
                             'img' to get only the image leg submodel, with only the image-embedding output.
        :return: 
        '''

        if submodel is not None and submodel not in ['txt', 'img']:
            ValueError('Value for submodel parameter not legal: ' + str(submodel) + '.'
                       '\nValue for submodel parameter must be "txt" or "img" or None.')

        # Image network leg:
        im_input = previous_tensor = Input(shape=(self.im_input_dim,), name='im_input')
        if im_hidden_layers is not None:
            for i, hidden_units in enumerate(im_hidden_layers):
                previous_tensor = Dense(output_dim=hidden_units, name='im_hidden_' + str(i))(previous_tensor)
                if im_hidden_activation is not None:
                    previous_tensor = Activation(activation=im_hidden_activation[i])(previous_tensor)
        im_emb = Dense(self.output_dim, activation=activation, name='im_embedding')(previous_tensor)
        im_emb = Activation(activation=activation)(im_emb)

        # Text network leg:
        tx_input = previous_tensor = Input(shape=(self.tx_input_dim,), name='tx_input')
        if tx_hidden_layers is not None:
            for i, hidden_units in enumerate(tx_hidden_layers):
                previous_tensor = Dense(output_dim=hidden_units, name='tx_hidden_' + str(i))(previous_tensor)
                if tx_hidden_activation is not None:
                    previous_tensor = Activation(activation=tx_hidden_activation[i])(previous_tensor)
        tx_emb = Dense(self.output_dim, activation=activation, name='tx_embedding')(previous_tensor)
        tx_emb = Activation(activation=activation)(tx_emb)

        #im_emb_sub_tx_emb = merge([im_emb, tx_emb], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0], name='subtract')
        #im_emb_sub_tx_emb = Merge(mode=euclideanSqDistance, output_shape=lambda x: (x[0][0], 1), name='distance')([im_emb, tx_emb])
        #im_emb_sub_tx_emb = Merge(mode=euclideanDistance, output_shape=lambda x: (x[0][0], 1), name='distance')([im_emb, tx_emb])
        #im_emb_sub_tx_emb = Merge(mode='cos')([im_emb, tx_emb])

        #im_emb_sub_tx_emb = merge([im_emb, tx_emb], mode='sum')
        #im_emb_sub_tx_emb = im_emb - tx_emb


        if submodel is not None:
            if submodel == "img":
                model = Model(input=im_input, output=im_emb)
                model.compile(optimizer=optimizer, loss=fake_loss)
            elif submodel == "txt":
                model = Model(input=tx_input, output=tx_emb)
                model.compile(optimizer=optimizer, loss=fake_loss)
            else:
                model = None
        else:
            model = Model(input=[im_input, tx_input], output=[im_emb, tx_emb]) #, im_emb_sub_tx_emb])
            model.summary()
            model.compile(optimizer=optimizer,
                          loss=[ get_contrastive_loss(tx_emb), get_contrastive_loss(im_emb),
                              #cos_distance #'mean_squared_error'#contrastive_loss
                          ],
                          # mse_on_sub_out_loss - da usare con 'subtract' merge invece di 'distance' merge
                          loss_weights=[1, 1]) #, -1 * tx_im_factr ])
        return model

def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))

def euclideanSqDistance(inputs):
    if (len(inputs) != 2):
        raise 'oops'
    output = K.mean(K.square(inputs[1] - inputs[0]), axis=-1)
    output = K.expand_dims(output, 1)

    return output



def euclideanDistance(inputs):
    if (len(inputs) != 2):
        raise 'oops'
    output = K.sqrt(K.sum(K.square(inputs[1] - inputs[0]), axis=-1))
    #output = K.expand_dims(output, 1)
    return output


def fake_loss(y_true, y_pred):
    import numpy as np
    return K.variable(np.array([0]))

def contrastive_loss_cos(labels, dists):  #cos merge return cosine similarity (1 when similars).

    label_first = labels[0:1, :]
    other_labels = labels[1:, :]

    labels_shifted = K.concatenate([labels, other_labels, label_first], axis=0) #   [ l1 ........ ln  | l2 ... ln-1 ln ]
    labels_orig = K.concatenate([labels, labels], axis=0)                       #   [ l1 ........ ln  | l1 ... ln-2 ln ]
    zeros = K.zeros_like(labels_orig)                                           #   [ 0  ........  0  | 0  ...   0   0 ]
    h = K.cast(K.equal(labels_orig-labels_shifted, zeros), dtype='float32')     #   [ 1  1 ......  1  | 0  ...   1   0 ]
                                                                                # h:   ALL ONES       |    MOST ZEROS
    # h[i] = 1  where labels_orig[i] == labels_shifted[i]  (i-th image correlated with i+1-th image, i.e. same artwork)
    # h[i] = 0  where labels_orig[i] != labels_shifted[i]

    first_dist = dists[0:1]
    other_dists = dists[1:]
    shifted_dists = K.concatenate([dists, other_dists, first_dist], axis=0) # [ d1 ........ dn  | d1 ... dn-2 dn ]

    # # EUCLIDEAN:
    # # equation:  Lcon = (1/2N) SUM[ h(i) d(i)^2 + (1-h(i)) max(1-d(i), 0)^2
    Z = K.zeros_like(shifted_dists)
    max_z_sd = K.max(K.stack([1-shifted_dists, Z]), axis=0, keepdims=False)
    #max_z_sd = K.sqrt(K.cast(K.shape(shifted_dists)[0], dtype='float32')) - shifted_dists

    first_operand = h*K.square(shifted_dists)
    second_operand = (1-h)* K.square(max_z_sd)


    # COSINE:
    # first_operand = h*(1-shifted_dists)
    # second_operand = (1-h)*(shifted_dists)



    tensor_sum = first_operand + second_operand
    sum = K.sum( tensor_sum, axis=0 )/K.cast(K.shape(shifted_dists)[0], dtype='float32')

    return K.mean(sum)


def get_contrastive_loss(text_outputs, margin=1):
    def contrastive_loss_2(labels, im_outputs):
        distances = K.sqrt(K.sum(K.square(im_outputs - text_outputs), axis=-1))

        first_text = text_outputs[0:1, :]
        last_texts = text_outputs[1:, :]
        shifted_texts = K.concatenate([last_texts, first_text], axis=0)

        shifted_distances =  K.sqrt(K.sum(K.square(im_outputs - shifted_texts), axis=-1))

        loss = K.mean( distances + K.maximum(margin-shifted_distances, 0))
        return loss
    return contrastive_loss_2


def contrastive_loss(labels, dists):

    label_first = labels[0:1, :]
    other_labels = labels[1:, :]

    labels_shifted = K.concatenate([labels, other_labels, label_first], axis=0) #   [ l1 ........ ln  | l2 ... ln-1 ln ]
    labels_orig = K.concatenate([labels, labels], axis=0)                       #   [ l1 ........ ln  | l1 ... ln-2 ln ]
    zeros = K.zeros_like(labels_orig)                                           #   [ 0  ........  0  | 0  ...   0   0 ]
    h = K.cast(K.equal(labels_orig-labels_shifted, zeros), dtype='float32')     #   [ 1  1 ......  1  | 0  ...   1   0 ]
                                                                                # h:   ALL ONES       |    MOST ZEROS
    # h[i] = 1  where labels_orig[i] == labels_shifted[i]  (i-th image correlated with i+1-th image, i.e. same artwork)
    # h[i] = 0  where labels_orig[i] != labels_shifted[i]

    first_dist = dists[0:1]
    other_dists = dists[1:]
    shifted_dists = K.concatenate([dists, other_dists, first_dist], axis=0) # [ d1 ........ dn  | d1 ... dn-2 dn ]


    # equation:  Lcon = (1/2N) SUM[ h(i) d(i)^2 + (1-h(i)) max(1-d(i), 0)^2
    Z = K.zeros_like(shifted_dists)
    max_z_sd = K.max(K.stack([1-shifted_dists, Z]), axis=0, keepdims=False)
    #max_z_sd = K.sqrt(K.cast(K.shape(shifted_dists)[0], dtype='float32')) - shifted_dists


    first_operand = h*K.square(shifted_dists)
    second_operand = (1-h)* K.square(max_z_sd)
    tensor_sum = first_operand + second_operand
    sum = K.sum( tensor_sum, axis=0 )/K.cast(K.shape(shifted_dists)[0], dtype='float32')

    return K.mean(sum)


def avg_batch_mse_loss_exaustive(y_true, y_pred):
    batch_size = K.int_shape(y_pred)[0]
    loss = 0
    N=0
    for i in range(0, batch_size):
        for j in range(i+1, batch_size):
            loss += mean_squared_error(y_pred[i], y_pred[j])
            N+=1
    loss /= N







def mse_on_sub_out_loss(y_true, y_pred):
    # NB: mean_squared_errror(a, b) = K.mean(K.square(a-b))
    # We use this mse_on_sub_output over the (im_emb-tx_emb) output of the network to compute MSE(im_emb, tx_emb)
    # return mean_squared_error(y_true, y_pred)
    return K.mean(K.square(y_pred))

def avg_batch_mse_loss(y_true, y_pred):
    # batch_size = K.int_shape(y_pred)[-1]
    # loss = 0
    # for i in range(0, batch_size):
    #     loss += mean_squared_error(y_pred[i], y_pred[i-1])
    # loss/=batch_size

    # max distance in the euclidean spaces with domain [0, 1] for each dimensionality, is the square root  of the number
    # of dimensions (1 for 1D, 1.414.. 2D, 1.73 3D, 2 4D....)

    y_pred_first_row = y_pred[0:1, :]
    y_pred_other_rows = y_pred[1:, :]
    y_pred_shifted =  K.concatenate([y_pred_other_rows, y_pred_first_row], axis=0)

    max_distance = np.sqrt(K.int_shape(y_pred)[1])
    return K.variable(np.array([max_distance])) - mean_squared_error(y_pred, y_pred_shifted)

    # mse = K.mean(K.sqrt(K.sum(K.square(y_pred - y_pred_shifted), axis=1)))
    # dist = K.variable(np.array([1])) - mse
    # zero = K.variable(np.array([0]))
    # concat = K.concatenate([zero, dist])
    # return K.max(concat)










































#
#
# def get_joint_embedding_loss(lmb1=1, lmb2=1):
#     def joint_embedding_loss(y_true, y_pred):
#         im_emb = y_pred[0]
#         tx_emb = y_pred[1]
#
#         shape = K.int_shape(y_pred)
#         if(shape[-1] == 4):
#             #if K.equal(shape,  K.variable(value=np.array([5]), dtype='int', name='shape_tensor')):
#             # TODO
#             lambda_tensor =  K.variable(value=np.array([[-lmb1], [lmb2], [lmb3], [-lmb4]]), dtype='int', name='lambda_tensor')
#             #print y_pred.eval()
#             y_pred_shape = K.int_shape(y_pred)
#             l_t_shape = K.int_shape(lambda_tensor)
#             ret = K.dot(y_pred, lambda_tensor)
#             return ret
#         else:
#             raise ValueError("joint_embedding_loss expect a vector of 4 elements")
#     return joint_embedding_loss


 #
 #
 # def siamese_model(self, optimizer,im_hidden_layers=None, tx_hidden_layers=None, activation='relu',
 #                      lambda1=1, lambda2=1, lambda3=0, lambda4=0):
 #
 #        # Image siamese network:
 #        im_input_1 = previous_tensor_1 = Input(shape=(self.im_input_dim,), name='im_input_1')  # name='im_input'
 #        im_input_2 = previous_tensor_2 = Input(shape=(self.im_input_dim,), name='im_input_2')  # name='im_input'
 #        if im_hidden_layers is not None:
 #            for i, hidden_units in enumerate(im_hidden_layers):
 #                hidden_layer = Dense(output_dim=hidden_units, name='im_hidden_' + i)
 #                previous_tensor_1 = hidden_layer(previous_tensor_1)
 #                previous_tensor_2 = hidden_layer(previous_tensor_2)
 #        im_embedding_layer = Dense(self.output_dim, activation=activation, name='im_embedding')
 #        im_emb_tensor_1 = im_embedding_layer(previous_tensor_1)
 #        im_emb_tensor_2 = im_embedding_layer(previous_tensor_2)
 #
 #        # Text siamese network:
 #        tx_input_1 = previous_tensor_1 = Input(shape=(self.tx_input_dim,), name='tx_input_1')  # name='tx_input'
 #        tx_input_2 = previous_tensor_2 = Input(shape=(self.tx_input_dim,), name='tx_input_2')  # name='tx_input'
 #        if tx_hidden_layers is not None:
 #            for i, hidden_units in enumerate(tx_hidden_layers):
 #                hidden_layer = Dense(output_dim=hidden_units, name='tx_hidden_' + i)
 #                previous_tensor_1 = hidden_layer(previous_tensor_1)
 #                previous_tensor_2 = hidden_layer(previous_tensor_2)
 #        tx_embedding_layer = Dense(self.output_dim, activation=activation, name='tx_embedding')
 #        tx_emb_tensor_1 = tx_embedding_layer(previous_tensor_1)
 #        tx_emb_tensor_2 = tx_embedding_layer(previous_tensor_2)
 #
 #
 #
 #        def mse_merge(tensor_list):
 #            # 1: Calcolo la media AVG dei vettori x[i]
 #            # 2: Faccio dist[i] = ||AVG - x[i]||_2
 #            # 3: MSE = sum( dist[i] )
 #            # TODO
 #            avg = tensor_list[0]
 #            for i in range(1, len(tensor_list)):
 #                avg += tensor_list[i]
 #            avg = avg / len(tensor_list)
 #
 #            mse = K.abs(tensor_list[0]-avg)
 #            for i in range(1, len(tensor_list)):
 #                mse += K.abs(tensor_list[i]-avg)
 #
 #            return mse
 #
 #
 #        t1_vs_t2 = merge([tx_emb_tensor_1, tx_emb_tensor_2], mode=mse_merge, output_shape=[1], name='mse_t1_vs_t2')    # we want to maximize this
 #        im1_vs_tx1 = merge([im_emb_tensor_1, tx_emb_tensor_1], mode=mse_merge, output_shape=[1], name='mse_im1_vs_t1')  # we want to minimize this
 #        im2_vs_tx2 = merge([im_emb_tensor_1, tx_emb_tensor_1], mode=mse_merge, output_shape=[1], name='mse_im2_vs_t2')  # we want to minimize this
 #        im1_vs_im2 = merge([im_emb_tensor_1, im_emb_tensor_2], mode=mse_merge, output_shape=[1], name='mse_im1_vs_im2')  # we want to maximize this?
 #
 #        out_tensor = merge([t1_vs_t2, im1_vs_tx1, im2_vs_tx2, im1_vs_im2], mode='concat', concat_axis=-1, name='out_tensor')
 #        a_shape = K.int_shape(t1_vs_t2)
 #        a_shape = K.int_shape(im1_vs_tx1)
 #        a_shape = K.int_shape(im2_vs_tx2)
 #        a_shape = K.int_shape(im1_vs_im2)
 #        out_shape = K.int_shape(out_tensor)
 #
 #
 #        model = Model(input=[im_input_1, im_input_2, tx_input_1, tx_input_2], output=out_tensor)
 #        #model = Model(inputs=[im_input_1, im_input_2, tx_input_1, tx_input_2], outputs=out_tensor)  # for keras 2 ?
 #
 #        model.compile(optimizer=optimizer, loss=get_joint_embedding_loss(lambda1, lambda2, lambda3, lambda4))
 #        return model

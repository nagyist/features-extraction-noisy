from keras.engine import Input, Model, merge, Merge
from keras.initializations import glorot_normal
from keras.layers import Dense, LSTM, K, Activation, Lambda
from keras.metrics import mean_squared_error, cosine_proximity, categorical_crossentropy
from keras.models import Sequential, load_model
import numpy as np



from keras.metrics import cosine_proximity

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

class JointEmbedderTriplet():

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

    def __init__(self, im_dim, tx_dim, out_dim, n_text_classes, use_merge_distance=False):
        self.im_input_dim = im_dim
        self.tx_input_dim = tx_dim
        self.output_dim = out_dim
        self.n_text_classes = n_text_classes
        self.use_merge_distance = use_merge_distance

    @classmethod
    def load_model(self, model_path, weight_path=None, contrastive_loss_margin=1):
        contrastive_loss_im = get_contrastive_loss_im(K.variable(np.asarray([[0]])))
        contrastive_loss_tx = get_contrastive_loss_tx(K.variable(np.asarray([[0]])))
        contrastive_loss_over_distance = get_contrastive_loss_over_distance(contrastive_loss_margin)
        contrastive_loss_contr_data = get_contrastive_loss_contr_data(K.variable(np.asarray([[0]])))
        model = load_model(filepath=model_path,
                           custom_objects={ 'contrastive_loss_im' : contrastive_loss_im,
                                            'contrastive_loss_tx': contrastive_loss_tx,
                                            'my_cosine_proximity' : my_cosine_distance,
                                            'contrastive_loss_over_distance': contrastive_loss_over_distance,
                                            'contrastive_loss_contr_data':  contrastive_loss_contr_data,
                                            'triplet_loss': get_triplet_loss(K.variable(np.asarray([[0]]))),
                                            'fake_loss': fake_loss} )
        if weight_path is not None:
            model.load_weights(weight_path)
        return model

    def model(self,
              optimizer,
              tx_activation=None,
              im_activation=None,
              im_hidden_layers=None,
              tx_hidden_layers=None,
              init='glorot_normal',
              triplet_loss_margins=1,
              triplet_loss_weights=1,
              configs=['tii']): #configs: < anchor, positive, negative >
        '''
        
        :param optimizer: 
        :param im_hidden_layers: a list of numbers/str, for each number a dense layer will be created, for each string 
                                 an activation layer will be created.
        :param tx_hidden_layers: a list of numbers/str, for each number a dense layer will be created, for each string 
                                 an activation layer will be created.
        :param submodel: None to get the whole model, with 3 outputs: text, image, and (text-image) embedding output. 
                             'txt' to get only the text leg submodel, with only the text-embedding output. 
                             'img' to get only the image leg submodel, with only the image-embedding output.
        :return: 
        '''
        # tii config!
        legal_configs = ['tii', 'tit', 'itt', 'iti', 'i-t', 't-i']
        # nb: i-t == iit where the anchor and the positive example are the same: ||i-i|| - ||i-t|| = -||i-t||
        # to do 'ii2t', 'ii2i', 'tt2i' 'tt2t'
        if not isinstance(triplet_loss_margins, list):
            lst = []
            for config in configs:
                lst.append(triplet_loss_margins)
            triplet_loss_margins = lst
        if not isinstance(triplet_loss_weights, list):
            lst = []
            for config in configs:
                lst.append(triplet_loss_weights)
            triplet_loss_weights = lst

        # Image network leg:
        previous_layer = first_im_layer = Lambda(function=lambda x: x)
        if im_hidden_layers is not None:
            for i, hid in enumerate(im_hidden_layers):
                if isinstance(hid, int):
                    previous_layer = Dense(output_dim=hid, name='im_hidden_' + str(i), init=init)(previous_layer)
                elif isinstance(hid, basestring):
                    previous_layer = Activation(activation=hid)(previous_layer)
        im_emb = Dense(self.output_dim, name='im_embedding')(previous_layer)
        im_emb = Activation(activation=im_activation)(im_emb)

        # Text network leg:
        previous_layer = first_tx_layer = Input(shape=(self.tx_input_dim,), name='tx_input')
        if tx_hidden_layers is not None:
            for i, hid in enumerate(tx_hidden_layers):
                if isinstance(hid, int):
                    previous_layer = Dense(output_dim=hid, name='tx_hidden_' + str(i), init=init)(previous_layer)
                elif isinstance(hid, basestring):
                    previous_layer = Activation(activation=hid)(previous_layer)
        tx_emb = Dense(self.output_dim, name='tx_embedding', init=init)(previous_layer)
        tx_emb = Activation(activation=tx_activation)(tx_emb)

        im_pos_input = Input(shape=(self.im_input_dim,), name='im_pos_input')
        im_pos_input_2 = Input(shape=(self.im_input_dim,), name='im2_pos_input')
        im_neg_input = Input(shape=(self.im_input_dim,), name='im_neg_input')
        tx_pos_input = Input(shape=(self.tx_input_dim,), name='tx_pos_input')
        tx_neg_input = Input(shape=(self.tx_input_dim,), name='tx_neg_input')
        im_pos_encoded = first_im_layer(im_pos_input)
        im_neg_encoded = first_im_layer(im_neg_input)
        tx_pos_encoded = first_tx_layer(tx_pos_input)
        tx_neg_encoded = first_tx_layer(tx_neg_input)

        merges = {}
        for config in configs:
            if config[0] == 't':
                anchor = tx_pos_encoded
            elif config[0] == 'i':
                anchor = im_pos_encoded

            if config[1] == 't':
                pos = tx_pos_encoded
            elif config[1] == 'i':
                pos = im_pos_encoded
            elif config[1] == '-':
                pos = anchor

            if config[2] == 't':
                neg = tx_neg_encoded
            elif config[2] == 'i':
                neg = im_neg_encoded

            merge = Merge(mode=triplet_distance, output_shape=lambda x: (x[0][0], 1))([anchor, pos, neg])
            merges[config] = merge


        outputs = [im_emb, tx_emb]
        losses = [fake_loss, fake_loss]
        loss_weights = [0, 0]
        for i, m in enumerate(merges.values()):
            outputs.append(m)
            losses.append(get_triplet_loss(triplet_loss_margins[i]))
            loss_weights.append(triplet_loss_weights[i])

        model = Model(input=[im_pos_input, im_neg_input, tx_pos_input, tx_neg_input], output=outputs)
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

        return model


def my_cosine_distance(tensors):
    if (len(tensors) != 2):
        raise 'oops'
    a = K.l2_normalize(tensors[0], axis=-1)
    b = K.l2_normalize(tensors[1], axis=-1)
    return 1-K.mean(a * b)


def my_distance_shape(list_tensor_shape):
    return list_tensor_shape[0][0]


def triplet_distance(tensors):
    if (len(tensors) != 3):
        raise 'oops'
    anchor = tensors[0]
    pos = tensors[1]
    neg = tensors[2]
    dist_p = K.sqrt(K.sum(K.square(anchor-pos), axis=-1))
    dist_n = K.sqrt(K.sum(K.square(anchor-neg), axis = -1))
    return dist_p-dist_n

def get_triplet_loss(margin=1):
    def triplet_loss(y_true, y_pred):
        return K.max(0, margin + y_pred)
    return triplet_loss

def my_distance(tensors):
    if (len(tensors) != 2):
        raise 'oops'
    t1 = tensors[0] - tensors[1]
    t2 = K.square(t1)
    t3 = K.sum(t2,axis=-1)
    dist = K.sqrt(t3)
    #dist = K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1))
    return dist

def get_contrastive_loss_over_distance(margin=1):
    def contrastive_loss_over_distance(labels, distances):
        '''
        :param labels: 1D tensor containing 0 or 1 for each example
        :param distances: 
        :return: 
        '''
        # loss = K.mean((distances + K.maximum(margin-shifted_distances, 0)))
        # loss = K.mean(distances - shifted_distances)

        loss = K.mean((labels)*K.square(distances) + (1-labels) * K.square(K.maximum(margin-distances, 0)))
        #loss_pos = (labels)*K.square(distances)
        #loss_neg = (1-labels) * K.square(K.maximum(margin-distances, 0))
        return loss
    return contrastive_loss_over_distance






def get_contrastive_loss_contr_data(other_output, margin=1):
    def contrastive_loss_contr_data(labels, output):
        distances = K.sqrt(K.sum(K.square(output - other_output), axis=0))
        #loss = K.mean((distances + K.maximum(margin-shifted_distances, 0)))
        #loss = K.mean(K.square(distances) + K.square(K.maximum(margin-shifted_distances, 0)))
        #loss = K.mean(distances - shifted_distances)
        loss = K.mean((labels)*K.square(distances) + (1-labels)*K.square(K.maximum(margin-distances, 0)))

        return loss
    return contrastive_loss_contr_data



def get_contrastive_loss_im(tx_output, margin=1):
    def contrastive_loss_im(labels, im_output):
        distances = K.sqrt(K.sum(K.square(im_output - tx_output), axis=-1))

        first_text = tx_output[0:1, :]
        last_texts = tx_output[1:, :]
        shifted_texts = K.concatenate([last_texts, first_text], axis=0)

        shifted_distances = K.sqrt(K.sum(K.square(im_output - shifted_texts), axis=-1))

        #loss = K.mean((distances + K.maximum(margin-shifted_distances, 0)))
        #loss = K.mean(K.square(distances) + K.square(K.maximum(margin-shifted_distances, 0)))
        #loss = K.mean(distances - shifted_distances)
        loss = K.mean(K.square(distances) + K.square(K.maximum(margin-shifted_distances, 0)))

        return loss
    return contrastive_loss_im


def get_contrastive_loss_tx(im_output, margin=1):
    def contrastive_loss_tx(labels, tx_output):
        distances = K.sqrt(K.sum(K.square(im_output- tx_output), axis=-1))

        first_text = tx_output[0:1, :]
        last_texts = tx_output[1:, :]
        shifted_texts = K.concatenate([last_texts, first_text], axis=0)

        shifted_distances = K.sqrt(K.sum(K.square(im_output - shifted_texts), axis=-1))

        # loss = K.mean((distances + K.maximum(margin-shifted_distances, 0)))
        # loss = K.mean(K.square(distances) + K.square(K.maximum(margin-shifted_distances, 0)))
        # loss = K.mean(distances - shifted_distances)
        loss = K.mean(K.square(distances) + K.square(K.maximum(margin - shifted_distances, 0)))

        return loss

    return contrastive_loss_tx






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

def contrastive_loss_old(labels, dists):

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
    y_pred_shifted = K.concatenate([y_pred_other_rows, y_pred_first_row], axis=0)

    return -K.mean(K.square(y_pred - y_pred_shifted))
    #return -K.sqrt(K.sum(K.square(y_pred - y_pred_shifted), axis=-1))

    # max_distance = np.sqrt(K.int_shape(y_pred)[1])
    # return K.variable(np.array([max_distance])) - mean_squared_error(y_pred, y_pred_shifted)

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

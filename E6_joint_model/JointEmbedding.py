from keras.engine import Input, Model, merge, Merge
from keras.layers import Dense, LSTM, K
from keras.metrics import mean_squared_error
from keras.models import Sequential


class JointEmbedder():

    # Model Idea:
    #
    # |   fc
    # | ------------->  | = t \
    # |  300 -> 128            \
    #                           >- | y' = i o t
    # |                        /
    # |   fc                  /
    # | ------------->  | = i
    # |  2K -> 128
    # |

    def __init__(self, im_input_dim, tx_input_dim, output_dim):
        self.im_input_dim = im_input_dim
        self.tx_input_dim = tx_input_dim
        self.output_dim = output_dim


    def im_embedding_tensor(self, im_hidden_layers=None, activation='relu'):
        def ret(previous_tensor):
            if im_hidden_layers is not None:
                for i, hidden_units in enumerate(im_hidden_layers):
                    previous_tensor = Dense(output_dim=hidden_units, name='im_hidden_'+i)(previous_tensor)
            im_embedding = Dense(self.output_dim, activation=activation)(previous_tensor)
            return im_embedding
        return ret

    def tx_embedding_tensor(self, tx_hidden_layers=None, activation='relu'):
        def ret(previous_tensor):
            if tx_hidden_layers is not None:
                for i, hidden_units in enumerate(tx_hidden_layers):
                    previous_layer = Dense(output_dim=hidden_units, name='im_hidden_'+i)(previous_layer)
            tx_embedding = Dense(self.output_dim, activation=activation)(previous_layer)
            return tx_embedding
        return ret

    # def model(self, optimizer, im_hidden_layers=None, tx_hidden_layers=None):
    #
    #     #dot_product = Merge([im_embedding, tx_embedding], mode='dot', dot_axes=(1, 1))
    #     import keras
    #     dot_product = merge(inputs=[tx_embedding, im_embedding], mode='dot')
    #
    #     model = Model(input=[im_input, tx_input], output=dot_product)
    #     model.compile(loss=joint_embedding_loss, optimizer=optimizer)
    #     return model


    def siamese_model(self, optimizer, lambda1=1, lambda2=1, im_hidden_layers=None, tx_hidden_layers=None, activation='relu'):

        # Image siamese network:
        im_input_1 = previous_tensor_1 = Input(shape=(self.im_input_dim,))  # name='im_input'
        im_input_2 = previous_tensor_2 = Input(shape=(self.im_input_dim,))  # name='im_input'
        if im_hidden_layers is not None:
            for i, hidden_units in enumerate(im_hidden_layers):
                hidden_layer = Dense(output_dim=hidden_units, name='im_hidden_' + i)
                previous_tensor_1 = hidden_layer(previous_tensor_1)
                previous_tensor_2 = hidden_layer(previous_tensor_2)
        im_embedding_layer = Dense(self.output_dim, activation=activation, name='im_embedding')
        im_emb_tensor_1 = im_embedding_layer(previous_tensor_1)
        im_emb_tensor_2 = im_embedding_layer(previous_tensor_2)

        # Text siamese network:
        tx_input_1 = previous_tensor_1 = Input(shape=(self.tx_input_dim,))  # name='tx_input'
        tx_input_2 = previous_tensor_2 = Input(shape=(self.tx_input_dim,))  # name='tx_input'
        if tx_hidden_layers is not None:
            for i, hidden_units in enumerate(tx_hidden_layers):
                hidden_layer = Dense(output_dim=hidden_units, name='tx_hidden_' + i)
                previous_tensor_1 = hidden_layer(previous_tensor_1)
                previous_tensor_2 = hidden_layer(previous_tensor_2)
        tx_embedding_layer = Dense(self.output_dim, activation=activation, name='tx_embedding')
        tx_emb_tensor_1 = im_embedding_layer(previous_tensor_1)
        tx_emb_tensor_2 = im_embedding_layer(previous_tensor_2)



        def mse_merge(tensor_list):
            # 1: Calcolo la media AVG dei vettori x[i]
            # 2: Faccio dist[i] = ||AVG - x[i]||_2
            # 3: MSE = sum( dist[i] )
            # TODO
            import numpy as np
            avg = K.mean(tensor_list)
            avg_val = K.eval(avg)
            avg_np_tensor = np.array(K.int_shape(tensor_list))
            avg_np_tensor[:] = avg_val
            avg_tensor = K.variable(avg_np_tensor)
            dist_tensor = K.abs(tensor_list-avg_tensor)
            mse = K.mean(dist_tensor)
            return mse


        t1_vs_t2 = merge([tx_emb_tensor_1, tx_emb_tensor_2], mode=mse_merge)    # we want to maximize this
        im1_vs_tx1 = merge([im_emb_tensor_1, tx_emb_tensor_1], mode=mse_merge)  # we want to minimize this
        im2_vs_tx2 = merge([im_emb_tensor_1, tx_emb_tensor_1], mode=mse_merge)  # we want to minimize this
        im1_vs_im2 = merge([im_emb_tensor_1, im_emb_tensor_2], mode=mse_merge)  # we want to maximize this?




        shared_base_model = self.model(optimizer=optimizer, im_hidden_layers=im_hidden_layers, tx_hidden_layers=tx_hidden_layers)


def get_joint_embedding_loss(lmb1=1, lmb2=1, lmb3=0, lmb4=0):
    mean_squared_error
    def joint_embedding_loss(y_true, y_pred):
        return y_pred*y_pred


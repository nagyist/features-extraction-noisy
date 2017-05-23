from keras.optimizers import Adadelta, Adam

from ExecutionConfig import Config




# opts = [Adadelta, Adam]
# bss = [8, 16, 32, 64, 128]
#
# lrs = [0.001, 0.01, 0.1,   1,  10, 100]
# eps =  [ 900,  900, 500, 300, 300, 200]
#
# sp_dims = [ 10, 50, 100, 200, 400, 600]
#
# tx_acts = ['sigmoid', 'tanh', 'softmax']
# im_acts = ['sigmoid', 'tanh', 'softmax']
#
#
# configs = []
# for opt in opts:
#  for bs in bss:
#   for lr, ep in zip (lrs, eps):
#    for sp_dim in sp_dims:
#                 configs.append(Config(lr=lr, opt=opt, bs=32, epochs=30, sp_dim=200,
#                                       tx_act='sigmoid', im_act='sigmoid',
#                                       im_hid=None, tx_hid=None,
#                                       contr_w=1, contr_inv_w=1, log_w=0, w_init='glorot_normal'))




def config_gen_ACTIVATIONS():
    # type: () -> (list[Config], basestring)
    im_acts = ['relu', 'sigmoid', 'softmax', 'tanh']
    tx_acts = ['relu', 'sigmoid', 'softmax', 'softmax',]
    sp_dims = [ 200 ]
    im_hids = [ None, [512, 'tanh'], [512, 'relu'], [512, 'relu'] ]
    tx_hids = [ None, [200, 'tanh'], [200, 'relu'], [200, 'tanh'] ]
    log_ws = [0, 1]
    # 32 CONFIGURATIONS!

    configs = []
    for tx_act, im_act in zip(tx_acts, im_acts):
        for sp_dim in sp_dims:
            for im_hid, tx_hid in zip(im_hids, tx_hids):
                for log_w in log_ws:

                    c = Config(lr=1, opt=Adadelta, bs=32, epochs=150,
                               sp_dim=sp_dim, tx_act=tx_act, im_act=im_act,
                               im_hid=im_hid, tx_hid=tx_hid,
                               contr_w=1, contr_inv_w=1, log_w_tx=log_w, w_init='glorot_normal')
                    configs.append(c)

    return configs, 'confgen-ACTIVATIONS'


def config_gen_DIMS():
    # type: () -> (list[Config], basestring)
    im_acts = ['sigmoid']
    tx_acts = ['sigmoid']
    sp_dims = [ 50, 100, 200, 400, 800 ]
    im_hids = [ None, [512, 'tanh'], [512, 'relu'], [512, 'relu'] ]
    tx_hids = [ None, [200, 'tanh'], [200, 'relu'], [200, 'tanh'] ]
    log_ws = [0, 1]
    # 50 CONFIGURATIONS!

    configs = []
    for tx_act, im_act in zip(tx_acts, im_acts):
        for sp_dim in sp_dims:
            for im_hid, tx_hid in zip(im_hids, tx_hids):
                for log_w in log_ws:

                    c = Config(lr=1, opt=Adadelta, bs=32, epochs=150,
                               sp_dim=sp_dim, tx_act=tx_act, im_act=im_act,
                               im_hid=im_hid, tx_hid=tx_hid,
                               contr_w=1, contr_inv_w=1, log_w_tx=log_w, w_init='glorot_normal')
                    configs.append(c)

    return configs, 'confgen-DIMS'

def config_gen_DIMS():
    # type: () -> (list[Config], basestring)
    im_acts = ['relu', 'sigmoid', 'softmax', 'tanh']
    tx_acts = ['relu', 'sigmoid', 'softmax', 'softmax',]
    sp_dims = [ 100, 200, 400]
    im_hids = [ None, [512, 'tanh'], [512, 'relu'], [512, 'relu'] ]
    tx_hids = [ None, [200, 'tanh'], [200, 'relu'], [200, 'tanh'] ]
    log_ws = [0, 1]

    configs = []
    for tx_act, im_act in zip(tx_acts, im_acts):
        for sp_dim in sp_dims:
            for im_hid, tx_hid in zip(im_hids, tx_hids):
                for log_w in log_ws:

                    c = Config(lr=1, opt=Adadelta, bs=32, epochs=150,
                               sp_dim=sp_dim, tx_act=tx_act, im_act=im_act,
                               im_hid=im_hid, tx_hid=tx_hid,
                               contr_w=1, contr_inv_w=1, log_w_tx=log_w, w_init='glorot_normal')
                    configs.append(c)

    return configs, 'confgen-DIMS'





def config_gen_TEST():
    # type: () -> (list[Config], basestring)
    im_acts = ['sigmoid']
    tx_acts = ['sigmoid']
    sp_dims = [200]
    # im_hids = [None, [512, 'tanh']]
    # tx_hids = [None, [200, 'tanh']]
    im_hids = [None]
    tx_hids = [None]

    # log_ws = [0, 1]
    log_ws = [0]

    configs = []
    for tx_act, im_act in zip(tx_acts, im_acts):
        for sp_dim in sp_dims:
            for im_hid, tx_hid in zip(im_hids, tx_hids):
                for log_w in log_ws:

                    c = Config(lr=10, opt=Adadelta, bs=64, epochs=2,
                               sp_dim=sp_dim, tx_act=tx_act, im_act=im_act,
                               im_hid=im_hid, tx_hid=tx_hid,
                               contr_w=1, contr_inv_w=1, log_w_tx=log_w, w_init='glorot_normal')
                    configs.append(c)

    return configs, 'TEST'








    # configs = []
    # c = Config()
    # c.lr = 10
    # c.opt = Adadelta
    # c.opt_str = 'adadelta'
    # c.bs = 64
    # c.epochs = 30
    # c.joint_space_dim = 200
    # c.tx_activation = 'sigmoid'
    # c.im_activation = 'sigmoid'
    # c.contrastive_loss_weight = 1
    # c.contrastive_loss_weight_inverted = 1
    # c.logistic_loss_weight = 0
    # c.weight_init = 'glorot_normal'
    # #c.weight_init = 'glorot_uniform' # 'glorot_normal'
    # #c.tx_hidden_layers = [200]
    # #c.tx_hidden_activation = ['relu']
    # #c.im_hidden_layers = [800]
    # #c.im_hidden_activation = ['relu']
    # # train_mAP-fit-end: 0.501253132832
    # # valid_mAP-fit-end: 0.501253132832
    # # test_mAP-fit-end: 0.505
    # # # ... in realta' abbiamo tutti i vettori delle distanze IDENTICI per questo si hanno questi risultati
    #
    # configs.append(c)

from E3_shallow.e2_finetune_labelflip import shallow_finetune_labelflip
from E3_shallow.e3_shallow_test import shallow_test
from config import cfg
from E3_shallow.e1_shallow_train import shallow_train

cfg.init()


#shallow_train()
#shallow_finetune_labelflip()
shallow_test()
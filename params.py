# REMEMBER TO CHANGE VERSION NUMBER
from models import ThreeConvGlobal, SixConv, SixConvPassThrough, SixConvPT_LFC, SixConvResidual

batch_size = 80
test_batch_size = 80
validation_split = .2
shuffle_dataset = False
random_seed = 37
dropout = True  # too much dropout?
learn_rate = .0007
intermediate_learn_rate = 0.002
lr_decay = 0.3
lr_cap_decay = 0.7
patience = 10
weight_decay = 0
epochs = 1000
version = '15d'
suppress_warnings = True
dataset = 'thous_pos'
model_type = SixConvResidual
interface_weight = .8
twohop = True
heads = 4

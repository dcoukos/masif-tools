# REMEMBER TO CHANGE VERSION NUMBER
from models import TwoConv, ThreeConv, FourConv, SixConv, EightConv, SixConvResidual, TwentyConv, TwentyConvPool, TwentyConvNoRes

batch_size = 2
test_batch_size = 2
validation_split = .10
shuffle_dataset = False
random_seed = 37
dropout = False  # too much dropout?
batchnorm = True
learn_rate = .0007
intermediate_learn_rate = 0.002
lr_decay = 0.3
lr_cap_decay = 0.7
patience = 10
weight_decay = 0
epochs = 200
version = '18'
suppress_warnings = True
dataset = 'thous'
model_type = SixConv
interface_weight = .8
twohop = True
heads = 4

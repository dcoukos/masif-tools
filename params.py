# REMEMBER TO CHANGE VERSION NUMBER
from models import ThreeConvBlock, TwentyConvNoRes, PretrainedBlocks

batch_size = 5
test_batch_size = 5
validation_split = .10
shuffle_dataset = False
random_seed = 37
dropout = False  # too much dropout?
batchnorm = True
learn_rate = .0015
intermediate_learn_rate = 0.002
lr_decay = 0.3
lr_cap_decay = 0.7
patience = 10
weight_decay = 0
epochs = 200
version = '20b_pos_full'
suppress_warnings = True
dataset = 'full'
model_type = PretrainedBlocks
interface_weight = .8
twohop = True
heads = 4

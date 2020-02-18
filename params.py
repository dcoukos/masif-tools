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
version = '20g_MFN'
suppress_warnings = True
dataset = 'thous'
<<<<<<< HEAD
model_type = None
=======
model_type = PretrainedBlocks
>>>>>>> ab0040fe50d823d44cd5576d0c32511df77eea8a
interface_weight = .8
twohop = True
heads = 4

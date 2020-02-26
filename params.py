# REMEMBER TO CHANGE VERSION NUMBER
from models import TwoConv, FourConv, SixConv, EightConv, TenConv, FourteenConv

batch_size = 1
test_batch_size = 1
validation_split = .10
shuffle_dataset = False
random_seed = 37
dropout = False  # too much dropout?
batchnorm = True
learn_rate = .003
intermediate_learn_rate = 0.002
lr_decay = 0.3
lr_cap_decay = 0.7
patience = 10
weight_decay = 0
epochs = 200
version = 'exp1-2conv-electro'
suppress_warnings = True
dataset = 'full_train_ds'
model_type = TwoConv
interface_weight = .8
twohop = True
heads = 4

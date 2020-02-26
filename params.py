# REMEMBER TO CHANGE VERSION NUMBER
from models import TwoConv, SixConv, TenConv, FourteenConv

batch_size = 1
test_batch_size = 1
validation_split = .10
shuffle_dataset = False
random_seed = 37
learn_rate = .001
intermediate_learn_rate = 0.002
lr_decay = 0.3
lr_cap_decay = 0.7
patience = 10
weight_decay = 0
epochs = 60
version = 'exp1_14conv-electro'
suppress_warnings = True
dataset = 'masif_site'
model_type = FourteenConv
interface_weight = .8
twohop = True
heads = 4

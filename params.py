# REMEMBER TO CHANGE VERSION NUMBER
from models import FourConv, TenConv, FourteenConv

batch_size = 1
test_batch_size = 1
validation_split = .10
shuffle_dataset = False
random_seed = 37
learn_rate = .00031
weight_decay = 0
epochs = 100
version = 'exp2_2hops'
suppress_warnings = True
dataset = 'masif_site'
model_type = FourConv
interface_weight = .8
twohop = True
heads = 4
coverage = 0.7
hops = 5

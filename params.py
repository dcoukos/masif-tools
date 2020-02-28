# REMEMBER TO CHANGE VERSION NUMBER
from models import SageNet, TwoConv, SixConv, TenConv, FourteenConv

batch_size = 1
test_batch_size = 1
validation_split = .10
shuffle_dataset = False
random_seed = 37
learn_rate = .00031
weight_decay = 0
epochs = 100
version = 'Neighborhood_Sampler'
suppress_warnings = True
dataset = 'named_masif'
model_type = SageNet
interface_weight = .8
twohop = True
heads = 4

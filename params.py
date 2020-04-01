# REMEMBER TO CHANGE VERSION NUMBER
from models import SixConv, TenConv, TenConvwRes, TenConvwAffine, FourteenConv, Spectral, MultiScaleEncoder

batch_size = 1
test_batch_size = 1
validation_split = .10
shuffle_dataset = False
random_seed = 37
learn_rate = .00030
weight_decay = 0
epochs = 50
version = 'exp_4_actual_relu'
suppress_warnings = True
dataset = 'masif_site'
model_type = FourteenConv
interface_weight = .8
twohop = True
heads = 4
hops = 5
coverage = 0.7
hops = 5
relu = False

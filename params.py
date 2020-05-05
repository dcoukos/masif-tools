# REMEMBER TO CHANGE VERSION NUMBER
from models import EightConv

batch_size = 1
test_batch_size = 1
validation_split = .10
shuffle_dataset = False
random_seed = 37
learn_rate = .00030
weight_decay = 0
epochs = 150
version = 'paper_depth_8conv'
suppress_warnings = True
dataset = 'masif_site'
model_type = EightConv
interface_weight = .8
twohop = True
heads = 4
coverage = 0.7

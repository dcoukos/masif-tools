from utils import generate_surface
import params as p

generate_surface(p.model_type, 'models/Jan23_14:40_15b/final.pt', '3BIK_A', False)

generate_surface(p.model_type, 'models/Jan23_14:40_15b/final.pt', '4ZQK_A', False)


import os

import glob
paths = glob.glob(os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/structures/*'))


test_paths = []
with open('./lists/testing.txt', 'r') as f:
    for line in f:
        line = line.split('\n')[0]
        test_paths.append(line)
test_paths
for name in test_paths:
    if len(name.split('_')) == 3:
        a, b, c = name.split('_')
        name = a + '_' + b
        structure2 = a + '_' + c
        test_paths.append(structure2)
    new_path = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/structures/test/') + name + \
        '.ply'
    path = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/structures/') + name + \
        '.ply'
    try:
        os.replace(path, new_path)
    except:
        print('{} not found'.format(name))

# rest done from command line.

import pathlib
pathlib.Path().absolute()

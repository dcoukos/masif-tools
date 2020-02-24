# masif-tools

### Designing amino acids around hotspots

**Todo:**
Dev Goal 1: develop model for predicting interface in ply files.

Experiment 1:
- Grid Search to find basic hyperparameters for network
- Parameters: {Depth of convolutional section, Input data}
- Depth: {2, 4, 6, 8, 10, 12... while perf increasing}
- Input Data: {Masif identifiers, Electrostatics, + Shape Index, +Rotated Positional Data}
- Uses SeLU because ReLU kills the learning...

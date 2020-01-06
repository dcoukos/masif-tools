# masif-tools

### Designing amino acids around hotspots

**Todo:**
    Dev Goal 1: develop model for predicting interface in ply files.

    Step 1: Basline Implementation

    [x] Batch import the dataset
    [x] Figure out train/validation split
    [x] Create the network --> Current
    [x] Write a training step
    [x] Train the model


    Step 2:
    [-] Reimplement network with Deep Graph Library
    [/] Create in-memory dataset, and mini-dataset. --> Current
    [] Create stored dataset
    [x] Set-up tensorboard reporting
    [/] Implement ROC AUC eval metric.
    [] Implement Weight Bias initializations.
    [x] Implement modifications to deal with unbalanced data
    [/] Train on full & mini dataset in cluster.
    [x] Analyzing why the model won't train
    [] Implement tensorboard hyperparameter tracking

    Next:
    - Increasing parameters
    - Include penalty for non-grouped predictions.
    - Implement FeaStNet network.
    - Introduce custom pooling algorithm which does avg on the charge & max on
        the surface.
    - Introduce multi-scale architecture
    - adaptive learning rate.
    - evaluate optimizers other than Adam?
    - Data parallelism
    -

    Then:
    - Study following networks and determine if they might be interesting:
        - Graph Attention network
        - Line Graph neural network
        - Tree-LSTM (graph batching)

    Finally:
    - Next challenge!! --> Discriminate synthetic surfaces.

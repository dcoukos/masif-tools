# masif-tools

### Designing amino acids around hotspots

Welcome to this repository. You'll find work I did for my master's project here.

## Summary
The study of protein-protein interfaces (PPIs) promises to open new avenues in the
generation of de novo protein binders. These molecules are essential in the design of protein drugs, or synthetic biological components. In this work, I investigated the use of graph neural networks for the purposes of identifying the binding sites of individual proteins normally in complex with other proteins. This work builds heavily on published work from members of the LPDI, where I did my master's thesis. Please check out their [paper](https://www.nature.com/articles/s41592-019-0666-6) about their framework **Masif** as and their [github repository](https://github.com/lpdi-epfl/masif).

### Organization of this repository
- baseline_run.py is the main run file for training models in this repository. Some
  models require different runfiles, and those can be found in ./experiments/ These files have to be moved up to the highest level of the repository before they will run correctly.
- dataset.py contains methods for creating the training and testing datasets from the
  raw data from Masif.
- models.py contains all the implementations of networks that I created and tested during
  my project
- params.py allows easy access to training parameters to speed experiment implementation.
- transforms.py contains functions for feature engineering.
- utils.py contains small functions, mostly for I/O or performing small mathematical
  operations.
- /runs/ contains tensorboard files that I used to evaluate the performance of my
  experiments. I generated an enormous amount of tensorboard files; most of them are not
  on the repository.


## This work
As protein surfaces can be represented as a cloud of discretized surface points, they can also be considered a mesh or a network. In essence, drawing connections between the nodes allows the network to perform classification based on the features of a node's neighbors. Graph neural networks use an attention mechanism to parse the number of neighbors of each node to a constant value. It also allows filters to be reused. Later in this readme I describe the difference between Euclidean and non-Euclidean data. The essential point is that euclidian data structures allow convolutional neural networks to be extremely powerful and memory efficient. Protein surface data are not Euclidean, but graph neural network use some tricks to try to apply convolutional methods to non-Euclidean data.

In my work, I tried to find a graph neural network that could classify protein interfaces as accurately as Masif, while avoiding the heavy preprocessing that Masif required. Unfortunately this ended up not being possible, Euclidean data structures are just too good. Without the regularity of the data, graph neural networks use too much memory to justify not using preprocessing like Masif uses. The result was run times that were 100x longer, with less accuracy.


## Why this work?
Why is this question biologically interesting? Why is this method interesting?

### Biological Motivation
Identifying PPIs is an interesting problem in biology because it is immediately applicable to the design of inhibitory protein drugs. Likely binding sites are also likely to be inhibiting sites if an unnatural partner binds to them. In addition, while there are energy functions which define the likelihood that a site is a binding site using biochemical parameters, these rules were determined experimentally. While that certainly validates them, they are likely to be missing important and inconspicuous relationships, which might be captured by modern computational methods. These relationships would almost certainly be captured by a functional classifier. Finally, using single-body interface classification is a useful proof-of-concept for later exciting design tasks. This is because a number of technical prerequisites for design could be validated by a classifier.

### Computational Motivation
Traditional methods for protein design involve a significant amount of hand-tuning structures selected by randomly replacing amino acids and optimizing an energy function. This leads to a relatively small portion of the solution space being actually searched, and does not make use of pattern recognition to design likely interfaces.

A quick word about convolutional neural networks and euclidian data structures. Convolutional neural networks have become the method of choice for classifying data structures by extracting pattern information locally. This methods are not immediately applicable to protein surfaces however, since the protein surface cannot be represented efficiently in a Euclidian domain. With Euclidian data, data are arranged within a regular structure; consider how pixels in an image are always equally spaced, are always surrounded by the same number of pixels (except at the edges), and the grid is fractal. These properties in the data allow deep convolutional neural networks to reuse parameters heavily by leveraging the statistical properties of stationarity, locality, and compositionality. This in turn allows them to achieve impressive reasoning capacity while being memory-efficient.

Protein surfaces, especially when represented by point clouds, are not euclidian. They do not have an intrinsic notion of distance and position (e.g. citation
networks) or may not be regularly discretized. It is therefore not possible to learn on these datasets while relying on the notions of Euclidean distance or regularity per se. Euclidean operations on non-Euclidean data are, in particular, not shift-invariant, and may not contain an intrinsic notion of locality. How, then, to take advantage of the tools provided by convolution without being able to rely on the shortcuts provided by Euclidian data structures?

With MASIF, Gainza et al. use the manifolds to abstract the space around each point into a 2D space. This allows them to construct an "image" of the features around each point on a protein surface and classify this point as interface or non-interface. This method had excellent results, ran on a very lightweight network, but required an immense amount of preprocessing (tens of hours on the supercomputer). There was also no guarantee that the solution thus found was optimal. As data continues to become available, there was an incentive to find a solution to the classification problem that would require less preprocessing.

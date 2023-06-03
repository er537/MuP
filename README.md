This repository contains a lightweight, custom implementation of MuParameterization as described here: https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks

$\mu$-Parameterization is a reparameterization of some of the hyperparameters in a Neural Network that allows optimizing hyperparameters on a small model and transferring these optimimal hyperameters to larger models

Hyper-parameters that can be transferred:
- Learning Rate (and Schedule)
- Initialization
- Optimizer params (eg Adam eps/beta)

Dimensions that can be varied:
- Width
- Depth

** The small model must be trained using the same batch size and total number of steps as the model you wish to transfer hyperparameters to **


This implementation is compatible with DDP and FSDP.

 

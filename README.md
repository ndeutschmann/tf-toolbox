# Machine learning toolbox

## Training

This module implements an `ExperimentManager` class that contains an `OptimizerManager` and a `ModelManager` object. The experiment manager can steer ML experiments, collecting hyperparameters from the optimizer and the model. The optimizer and model managers are defined as abstract classes which must be implemented for specific model classes.
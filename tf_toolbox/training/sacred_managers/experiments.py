from sacred import Experiment
from ..experiment_manager import LoggingExperimentManager

# TODO:
# We want to provide several levels of abstraction.
# Both LoggingExperimentManager and TBExperimentManager can already be wrapped in a user defined script
# as seen in https://gist.github.com/ndeutschmann/b538c3b58cc4ae53a0e0e9df4c25ba85
# This would be enough for running myself, but it will be annoying to redo the same kind of thing for each
# type of experiment.
#
# Level one: very basic janitorial things like
#   - handling observer setting through the experiment manager to make all logging properly organized
#   - ensure consistent definition of the model yaml and hd5 info as artifacts
#   - other stuff
#
# Level two: fully inclusive package:
#   - treat the sacred experiment as an attribute
#   - generate the @main and @config functions automatically

class BasicSacredExperiment(LoggingExperimentManager):
    """TODO"""
    pass

class SacredExperiment(LoggingExperimentManager):
    """TODO"""
    pass

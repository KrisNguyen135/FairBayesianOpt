from gpflowopt.acquisition.acquisition import Acquisition
from gpflowopt.pareto import Pareto

from gpflow.param import DataHolder
from gpflow import settings

import numpy as np
import tensorflow as tf


stability = settings.numerics.jitter_level
float_type = settings.dtypes.float_type


class HVExpectedImprovement(Acquisition):
    def __init__(self, models):
        '''
        :param models: a list of (possibly multioutput) GPflow models representing
        our belief about the objectives.
        '''
        super(HVExpectedImprovement, self).__init__(models)
        num_objectives = self.data[1].shape[1]
        assert num_objectives > 1

        # Keep empty for now, will be updated in _setup()
        self.pareto = Pareto(np.empty((0, num_objectives)))
        self.reference = DataHolder(np.ones((1, num_objectives)))

    def _estimate_reference(self):
        pf = self.pareto.front.value
        f = np.max(pf, axis=0, keepdims=True) - np.min(pf, axis=0, keepdims=True)
        return np.max(pf, axis=0, keepdims=True) + 2 * f / pf.shape[0]

    def _setup(self):
        '''
        Pre-computes the Pareto set and cell bounds for integrating over the
        non-dominated regions.
        '''
        super(HVExpectedImprovement, self)._setup()

        # Obtain the hypervolume cell bounds, use prediction mean
        feasible_samples = self.data[0][self.highest_parent.feasible_data_index(), :]
        F = np.hstack((m.predict_f(feasible_samples)[0] for m in self.models))
        self.pareto.update(F)

        # Calculate reference point
        self.reference = self._estimate_reference()

    def build_acquisition(self, Xcand):
        outdim = tf.shape(self.data[1])[1]
        num_cells = tf.shape(self.pareto.bounds.lb)[0]
        N = tf.shape(Xcand)[0]  # number of candidates

        # Predictions for candidates, concatenate columns
        preds = [m.build_predict(Xcand) for m in self.models]
        candidate_mean, candiate_var = (tf.concat(moment, 1) for moment in zip(*preds))
        candidate_var = tf.maximum(candiate_var, stability)  # to avoid zeros

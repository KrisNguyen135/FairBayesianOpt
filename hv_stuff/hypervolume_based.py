import skopt
from skopt.plots import plot_convergence
import sklearn
import numpy as np
import matplotlib.pyplot as plt

import copy
import inspect
import numbers
from collections.abc import Iterable
from scipy.optimize import fmin_l_bfgs_b

import warnings
import joblib


def custom_gp_cook_estimator(space=None, **kwargs):
    '''
    Precondition: base estimator must be a GP regressor.
    '''

    if space is not None:
        space = skopt.space.space.Space(space)
        space = skopt.space.space.Space(skopt.utils.normalize_dimensions(
            space.dimensions))
        n_dims = space.transformed_n_dims
        is_cat = space.is_categorical
    else:
        raise ValueError('Expected a Space instance, not None')

    cov_amplitude = skopt.learning.gaussian_process.kernels.ConstantKernel(
        1.0, (0.01, 1000.0))
    # only special if all dimensions are categorical
    if is_cat:
        other_kernel = skopt.learning.gaussian_process.kernels.HammingKernel(
            length_scale=np.ones(n_dims))
    else:
        other_kernel = skopt.learning.gaussian_process.kernels.Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(0.01, 100)] * n_dims,
            nu=2.5
        )

    base_estimator = skopt.learning.gaussian_process.gpr.GaussianProcessRegressor(
        kernel=cov_amplitude * other_kernel,
        normalize_y=True,
        n_restarts_optimizer=2
    )

    base_estimator.set_params(**kwargs)
    return base_estimator


# def custom_gaussian_acquisition(X, model, acq_func, y_opt=None, return_grad=False,
#                                 acq_func_kwargs=None):
#     '''
#     Wrapper so that the output of this function can be directly passed to a
#     minimizer.
#     '''
#     # Check inputs
#     X = np.asarray(X)
#     if X.ndim != 2:
#         raise ValueError(f'X is {X.ndim}-dimensional, must be 2-dimensional')
#
#     if acq_func_kwargs is None:



class CustomOptimizer:
    def __init__(self, dimensions, acq_func, base_estimator='gp',
                 n_random_starts=None, n_initial_points=10, acq_optimizer='auto',
                 random_state=None, model_queue_size=None, acq_func_kwargs=None,
                 acq_optimizer_kwargs=None):
        self.rng = sklearn.utils.check_random_state(random_state)

        ### Configure acquisition function
        # Store and create acquisition function set
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get('eta', 1.0)

        ### Configure counters of points
        if n_random_starts is not None:
            n_initial_points = n_random_starts

        self._n_initial_points = n_initial_points
        self.n_initial_points = n_initial_points

        ### Configure estimator
        # Build base estimator if does not exist
        if isinstance(base_estimator, str):
            base_estimator = custom_gp_cook_estimator(
                space=dimensions,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max)
            )

        self.base_estimator_ = base_estimator

        ### Configure optimizer
        # Decide optimizer based on gradient information
        if acq_optimizer != 'sampling' \
                and skopt.utils.has_gradients(self.base_estimator_):
            acq_optimizer = 'lbfgs'
        else:
            acq_optimizer = 'sampling'
        self.acq_optimizer = acq_optimizer

        # Record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get('n_points', 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get(
            'n_restarts_optimizer', 5)
        self.n_jobs = acq_optimizer_kwargs.get('n_jobs', 1)
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        ### Configure search space
        # Normalize space
        dimensions = skopt.utils.normalize_dimensions(dimensions)
        self.space = skopt.space.space.Space(dimensions)

        # Record categorical and non-categorical indices
        self._cat_inds = []
        self._non_cat_inds = []
        for ind, dim in enumerate(self.space.dimensions):
            if isinstance(dim, skopt.space.space.Categorical):
                self._cat_inds.append(ind)
            else:
                self._non_cat_inds.append(ind)

        # Initialize storage for optimization
        self.max_model_queue_size = model_queue_size
        self.models = []
        self.Xi = []
        self.yi = []

        # Initialize cache for `ask` method responses
        self.cache_ = {}

    def copy(self, random_state=None):
        optimizer = CustomOptimizer(
            dimensions=self.space.dimensions,
            base_estimator=self.base_estimator_,
            n_initial_points=self.n_initial_points_,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state,
        )

        if hasattr(self, "gains_"):
            optimizer.gains_ = np.copy(self.gains_)

        if self.Xi:
            optimizer._tell(self.Xi, self.yi)

        return optimizer

    def ask(self):
        if self._n_initial_points > 0 or self.base_estimator_ is None:
            return self.space.rvs(random_state=self.rng)[0]

        if not self.models:
            raise RuntimeError('Random evaluations exhausted and with no fit models')

        next_x = self._next_x  # computed from last call to tell()
        min_delta_x = min([self.space.distance(next_x, xi)
                           for xi in self.Xi])
        if abs(min_delta_x) <= 1e-8:
            warnings.warn('The objective has been evaluated at this point before.')

        return next_x

    def tell(self, x, y, fit=True):
        skopt.utils.check_x_in_space(x, self.space)
        # self._check_y_is_valid(x, y)

        self.Xi.append(x)
        self.yi.append(y)
        self._n_initial_points -= 1

        # Optimzier learned something new - discard cache
        self.cache_ = {}

        # After being "told" `n_initial_points`, we switch from random sampling
        # to using a surrogate model
        if fit and self._n_initial_points <= 0 and self.base_estimator_ is not None:
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = sklearn.base.clone(self.base_estimator_)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # TODO: extend to MO
                est.fit(self.space.transform(self.Xi), self.yi)

            if self.max_model_queue_size is not None \
                    and len(self.models) >= self.max_model_queue_size:
                self.models.pop(0)
            # TODO: make `self.models` into a 2D matrix to extend to MO
            self.models.append(est)

            # Even with BFGS as the optimizer, we want to sample a large number
            # of points and then pick the best ones as starting points
            X = self.space.transform(self.space.rvs(
                n_samples=self.n_points, random_state=self.rng
            ))

            # TODO: extend to MO
            values = skopt.acquisition._gaussian_acquisition(
                X=X, model=est, y_opt=np.min(self.yi),
                acq_func=self.acq_func, acq_func_kwargs=self.acq_func_kwargs
            )

            # Find the minimum of the acquisition function by randomly
            # sampling points from the space
            if self.acq_optimizer == 'sampling':
                next_x = X[np.argmin(values)]

            elif self.acq_optimizer == 'lbfgs':
                # print('Using L-BFGS')
                x0 = X[np.argsort(values)[: self.n_restarts_optimizer]]

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    results = joblib.Parallel(n_jobs=self.n_jobs)(
                        joblib.delayed(fmin_l_bfgs_b)(
                            skopt.acquisition.gaussian_acquisition_1D, x,
                            args=(est, np.min(self.yi), self.acq_func,
                                  self.acq_func_kwargs),
                            bounds=self.space.transformed_bounds,
                            approx_grad=False,
                            maxiter=20
                        ) for x in x0
                    )

                cand_xs = np.array([r[0] for r in results])
                cand_acqs = np.array([r[1] for r in results])
                next_x = cand_xs[np.argmin(cand_acqs)]

                # lbfgs should be able to handle this but in case there are
                # prevision errors
                if not self.space.is_categorical:
                    next_x = np.clip(
                        next_x,
                        transformed_bounds[:, 0], transformed_bounds[:, 1]
                    )

            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1))
            )[0]

        return skopt.utils.create_result(self.Xi, self.yi, self.space, self.rng,
                                         models=self.models)

    # def _check_y_is_valid(self):
    #     return

    def run(self, func, n_iter=1):
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x))

        return skopt.utils.create_result(self.Xi, self.yi, self.space, self.rng,
                                         models=self.models)


def custom_base_minimize(func, dimensions, acq_func, base_estimator, n_calls=100,
                         n_random_starts=10, acq_optimizer='lbfgs', x0=None,
                         y0=None, random_state=None, verbose=False, callback=None,
                         n_points=10000, n_restarts_optimizer=5, xi=0.01,
                         n_jobs=1, model_queue_size=None):
    specs = {
        'args': copy.copy(inspect.currentframe().f_locals),
        'function': inspect.currentframe().f_code.co_name
    }

    acq_optimizer_kwargs = {
        'n_points': n_points,
        'n_restarts_optimizer': n_restarts_optimizer,
        'n_jobs': n_jobs
    }
    acq_func_kwargs = {'xi': xi}

    ### Initialize optimization
    # Suppose there are points provided (x0, y0), record them

    # Check x0
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]

    # Check y0
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]

    required_calls = n_random_starts + (len(x0) if not y0 else 0)
    if n_calls < required_calls:
        raise ValueError(f'Expected `n_calls` >= {required_calls}, got {n_calls}')
    n_initial_points = n_random_starts + len(x0)

    ### Build optimizer
    # Create optimizer object
    optimizer = CustomOptimizer(
        dimensions, acq_func, base_estimator=base_estimator,
        n_initial_points=n_initial_points, acq_optimizer=acq_optimizer,
        random_state=random_state, model_queue_size=model_queue_size,
        acq_optimizer_kwargs=acq_optimizer_kwargs, acq_func_kwargs=acq_func_kwargs
    )

    # Check x0: element-wise data type, dimensionality
    assert all(isinstance(p, Iterable) for p in x0)
    if not all(len(p) == optimizer.space.n_dims for p in x0):
        raise RuntimeError(f'Optimization space {optimizer.space} and initial points in x0 have inconsistent dimensions.')

    # Check callback
    callbacks = skopt.callbacks.check_callback(callback)
    if verbose:
        callbacks.append(skopt.callbacks.VerboseCallback(
            n_init=len(x0) if not y0 else 0,
            n_random=n_random_starts,
            n_total=n_calls
        ))

    ### Record provided points
    # Create return object
    result = None

    # Evaluate y0 if only x0 is provided
    if x0 and y0 is None:
        y0 = list(map(func, x0))
        n_calls -= len(y0)

    # Record through tell function
    if x0:
        if not isinstance(y0, Iterable) and not isinstance(y0, numbers.Number):
            raise ValueError(f'`y0` should be an iterable or scalar, got {type(y0)}')

        if len(x0) != len(y0):
            raise ValueError('`x0` and `y0` should have the same length')

        result = optimizer.tell(x0, y0)
        result.specs = specs
        if skopt.utils.eval_callbacks(callbacks, result):
            return result

    # Optimize
    for n in range(n_calls):
        next_x = optimizer.ask()
        next_y = func(next_x)
        result = optimizer.tell(next_x, next_y)
        result.specs = specs
        if skopt.utils.eval_callbacks(callbacks, result):
            break

    return result

def custom_gp_minimize(func, dimensions, acq_func, base_estimator=None,
                       n_calls=100, n_random_starts=10, acq_optimizer='lbfgs',
                       x0=None, y0=None, random_state=None, verbose=False,
                       callback=None, n_points=10000, n_restarts_optimizer=5,
                       xi=0.01, n_jobs=1, model_queue_size=None):
    rng = sklearn.utils.check_random_state(random_state)
    space = skopt.utils.normalize_dimensions(dimensions)

    if base_estimator is None:
        base_estimator = custom_gp_cook_estimator(
            space=space,
            random_state=rng.randint(0, np.iinfo(np.int32).max)
        )

    return custom_base_minimize(
        func, space, base_estimator=base_estimator, acq_func=acq_func,
        xi=xi, acq_optimizer=acq_optimizer, n_calls=n_calls, n_points=n_points,
        n_random_starts=n_random_starts, n_restarts_optimizer=n_restarts_optimizer,
        x0=x0, y0=y0, random_state=rng, verbose=verbose, callback=callback,
        n_jobs=n_jobs, model_queue_size=model_queue_size
    )


def hv_probability_of_improvement(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    return


def f(x):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))


if __name__ == '__main__':
    x = np.linspace(-2, 2, 400).reshape(-1, 1)
    fx = [f(x_i) for x_i in x]

    res = custom_gp_minimize(
        f, [(-2.0, 2.0)],
        acq_func='EI',
        n_calls=15,
        n_random_starts=5,
        random_state=123
    )

    print(res.x[0], res.fun)
    plot_convergence(res)
    plt.show()

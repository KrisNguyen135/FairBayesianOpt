import numpy as np
import scipy


NUM_POINTS_FRONTIER = 10
USE_GRID_ONLY = False
GRID_SIZE = 1000
NSGA_POP = 100
NSGA_EPOCHS = 100

SMSEGO_OPTION_DEFAULTS = {
    'ehi_pareto_set_size': 10,
    'ehi_grid_size': 1000,
    'ehi_nsga_epochs': 100,
    'ehi_nsga_pop': 100,
    'ehi_use_grid_only_to_solve_problem': False
}


class EHI:
    def __init__(self, num_dims, verbose=True,
                 input_space=None, grid=None, opt=None):
        global NUM_POINTS_FRONTIER
        global USE_GRID_ONLY
        global GRID_SIZE
        global NSGA_POP
        global NSGA_EPOCHS

        self.cached_info = dict()
        self.has_gradients = False
        self.num_dims = num_dims
        self.input_space = input_space

        self.options = SMSEGO_OPTION_DEFAULTS.copy()
        self.options.update(opt)

        NUM_POINTS_FRONTIER = self.options['ehi_pareto_set_size']
        GRID_SIZE = self.options['ehi_grid_size']
        USE_GRID_ONLY = self.options['ehi_use_grid_only_to_solve_problem']
        NSGA_POP = self.options['ehi_nsga_pop']
        NSGA_EPOCHS = self.options['ehi_nsga_epochs']

    def acquisition(self, obj_model_dict, con_models_dict, cand, current_best,
                    compute_grad, minimize=True, tasks=None):
        models = obj_model_dict.values()

        assert(
            len({model.state for model in models}) == 1,
            'Model are not all at the same state'
        )
        assert(not compute_grad)

        key = typle([obj_model_dict[obj].state for obj in obj_model_dict])

        if key not in self.cached_info:
            self.cached_info[key] = self.compute_cell_info(obj_model_dict)

        return self.compute_acquisition(cand, obj_model_dict, self.cached_info[key])

    def compute_cell_info(self, obj_model_dict):
        cached_info = dict()

        # Obtain a sample from the Pareto frontier
        moop

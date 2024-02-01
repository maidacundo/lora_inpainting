# class for scheduling timesteps between a list of values
import random

class TimestepScheduler():
    def __init__(self, change_every_n_steps=100, fixed_bounds=None):
        self.timesteps_dict = {
            0: [800, 600],
            1: [600, 500],
            2: [500, 400],
            3: [400, 200],
            4: [200, 1],
        }
        self.change_every_n_steps = change_every_n_steps
        self.current_step = 0
        self.current_idx = 0
        self.current_bounds = self.timesteps_dict[self.current_idx]
        if fixed_bounds is not None:
            self.fixed_bounds = fixed_bounds
        else:
            self.fixed_bounds = None

    def get_timesteps_bounds(self):
        if self.fixed_bounds is not None:
            return self.fixed_bounds
        
        if self.current_step % self.change_every_n_steps == 0:
            self.current_idx += 1
            if self.current_idx >= len(self.timesteps_dict):
                self.current_idx = 0 
            self.current_bounds = self.timesteps_dict[self.current_idx] 
            print(f"New timesteps bounds: {self.current_bounds[0]} - {self.current_bounds[1]}")
        self.current_step += 1
        
        return self.current_bounds
# class for scheduling timesteps between a list of values
import random

class TimestepScheduler():
    def __init__(self, change_every_n_steps=100):
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

    def get_timesteps_bounds(self):
        if self.current_step % self.change_every_n_steps == 0:
            self.current_idx += 1
            if self.current_idx >= len(self.timesteps_dict):
                self.current_idx = 0 
            self.current_bounds = self.timesteps_dict[self.current_idx] 
            print(f"New timesteps bounds: {self.current_bounds[0]} - {self.current_bounds[1]}")
        self.current_step += 1
        return self.current_bounds
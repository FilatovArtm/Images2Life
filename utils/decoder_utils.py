from utils.common_utils import numpyToVar
import torch
import numpy as np

"""
Utils for exepriment where 1d vector with time and picture component is transformed into video.
"""


class SpatialVectorGenerator:
    def __init__(self, time_size, picture_size, noise_level=0.1):
        self.n_examples = n_examples
        self.video_length = video_length
    
        self.code_generators = []
        for i in range(self.n_examples):
            self.code_generators.append(SpatialVectorGenerator(config['time_code_size'],config['pic_code_size']))
            
        for i in range(1, self.n_examples):
            self.code_generators[i].variables_["time_gamma"] = self.code_generators[0].variables_["time_gamma"]
            self.code_generators[i].variables_["time_delta"] = self.code_generators[0].variables_["time_delta"]

    def __call__(self, start_T, end_T, k=0, r=0):
        res = torch.sin(torch.ger(torch.arange(start_T, end_T), self.variables_["time_gamma"]) + \
            self.variables_["time_delta"])
        pic = self.variables_["picture"].expand(end_T - start_T, len(self.variables_["picture"]))
        return torch.cat([pic, res], dim=1)

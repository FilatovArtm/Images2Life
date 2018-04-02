"""
Utils for exepriment where 1d vector with time and picture component is transformed into video.
"""


class SpatialVectorGenerator:
    def __init__(self, time_size, picture_size, noise_level=0.1):
        self.variables_ = {"picture": None, "time_gamma": None, "time_delta": None}
        self.variables_["picture"] = numpyToVar(np.random.normal(
                0, noise_level, picture_size), requires_grad=True)
        self.variables_["time_gamma"]  = numpyToVar(np.random.normal(
                0, noise_level, time_size), requires_grad=True)
        self.variables_["time_delta"]  = numpyToVar(np.random.normal(
                0, noise_level, time_size), requires_grad=True)

    def __call__(self, start_T, end_T, k=0, r=0):
        res = torch.sin(torch.ger(torch.arange(start_T, end_T).cuda(), self.variables_["time_gamma"]) + self.variables_["time_delta"])
        pic = self.variables_["picture"].expand(end_T - start_T, len(self.variables_["picture"]))
        return torch.cat([pic, res], dim=1)

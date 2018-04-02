"""
Utils for experiment where net is simultaniously trained to predict the video and image.
"""

class BatchGeneratorVideoAndImage:
    def __init__(self, target, maps_generator_video, maps_generator_image, k, r):
        self.target_ = target
        self.maps_generator_video_ = maps_generator_video
        self.maps_generator_image_ = maps_generator_image
        self.k = k
        self.r = r
        self.T = len(target)

    def __call__(self, mode='train'):
        if mode == 'train':
            return torch.cat([self.maps_generator_video_(0, len(self.target_) - 1, self.k, self.r), 
                              self.maps_generator_image_(0, 1, self.k, self.r)]), self.target_
        else:
            return self.maps_generator_image_(0, 64, self.k, self.r)

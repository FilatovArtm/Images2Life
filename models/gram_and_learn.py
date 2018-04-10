import torch
import torch.utils.data
import numpy as np
from skimage.transform import resize
from utils.common_utils import generateSyntheticTexture, \
                         generateSyntheticImage, \
                         preprocessTarget, \
                         numpyToVar


class TimeMapsGenerator:

    def __init__(self, M, noise_level=0.1):
        self.spatial_variables = {"gamma": None, "delta": None}
        self.maps_number_ = M
        for key in self.spatial_variables:
            self.spatial_variables[key] = numpyToVar(np.random.normal(
                0, noise_level, M), requires_grad=True)

    def __call__(self, size, time_stamp):
        single_map = torch.zeros((self.maps_number_, size, size))
        single_map += self.spatial_variables["delta"][:, None, None]

        single_map += (self.spatial_variables["gamma"]
                       * time_stamp)[:, None, None]
        return single_map


class GramAndImages(torch.utils.data.TensorDataset):
    def __init__(self, video_length, video_seed, maps_number, size):
        self.size_ = size
        self.video_length_ = video_length
        self.video_ = generateSyntheticTexture(random=True, seed=video_seed)

        # resize and transpose
        self.video_ = preprocessTarget(self.video_, video_length, size)
        self.maps_generator_ = TimeMapsGenerator(maps_number)

    def __getitem__(self, index):
        '''
        One unit contains one image of video and one picture
        '''
        image = generateSyntheticImage(index)[0]
        image = np.transpose(image, [2, 0, 1])
        image = numpyToVar(
            resize(image, output_shape=(3, self.size_, self.size_), mode='constant'))
        # resize and transpose

        video_frame = self.video_[index]

        input_im = torch.cat(
            [image, self.maps_generator_(self.size_, time_stamp=0)])
        input_vid = torch.cat(
            [video_frame, self.maps_generator_(self.size_, time_stamp=index)])

        return torch.cat([input_im[None], input_vid[None]]), torch.cat([image[None], video_frame[None]])

    def __len__(self):
        return self.video_length_

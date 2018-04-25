import torch
import torch.utils.data
import numpy as np
from skimage.transform import resize
from utils.common_utils import generateSyntheticTexture, \
                         generateSyntheticImage, \
                         preprocessTarget, \
                         numpyToVar
import skvideo

def loss(y_hat, y):
    result = 0
    for i in range(len(y)):
        if i % 2 == 0:
            gram_y = y[i] @ torch.transpose(y[i], 2, 1)
            gram_yhat = y_hat[i] @ torch.transpose(y_hat[i], 2, 1)
            result += torch.sum((gram_y - gram_yhat) ** 2) / (gram_y.shape[1] ** 2) / 3
        else:
            result += torch.sum((y[i] - y_hat[i]) ** 2) / (y.shape[1] ** 2) / 3
            
    return result / len(y)


            

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
    def __init__(self, video_length, maps_number, size, video_seed=0, video_path=None):
        
        self.size_ = size
        self.video_length_ = video_length
        self.video_path_ = video_path
        
        if video_path == None:
            self.video_ = generateSyntheticTexture(random=True, seed=video_seed)
        else:
            self.full_video_ = skvideo.io.vread(fname=video_path)
            self.video_ = skvideo.io.vread(fname=video_path)[:, :size, :size]
            
        # resize and transpose
        self.video_ = preprocessTarget(self.video_, video_length, size)
        self.maps_generator_ = TimeMapsGenerator(maps_number)

    def __getitem__(self, index):
        '''
        One unit contains one image of video and one picture
        '''
        if self.video_path_ == None:
            image = generateSyntheticImage(index)[0]
        else:
            np.random.seed(index)
            coord = np.random.choice(len(self.full_video_[0]) - self.size_, size=2)
            image = self.full_video_[index, coord[0]:coord[0] + self.size_, coord[1]:coord[1] + self.size_]
        image = np.transpose(image, [2, 0, 1])
        image = numpyToVar(
            resize(image, output_shape=(3, self.size_, self.size_), mode='constant'))
        # resize and transpose

        video_frame = self.video_[index]

        np.random.seed(index)
        time_stamp = np.random.choice(self.video_length_)
        input_im = torch.cat(
            [image, self.maps_generator_(self.size_, time_stamp=time_stamp)])
        input_vid = torch.cat(
            [self.video_[0], self.maps_generator_(self.size_, time_stamp=index)])

        return torch.cat([input_im[None], input_vid[None]]), torch.cat([image[None], video_frame[None]])

    def __len__(self):
        return self.video_length_
    
    

    
class GramAndImagesTest(torch.utils.data.TensorDataset):
    def __init__(self, video_length, video_seed, maps_number, size, maps_generator, origin_frame=None):
        
        self.size_ = size
        self.video_length_ = video_length

        # resize and transpose
        if origin_frame is None:
            self.image_ = generateSyntheticImage(video_seed + 1000)
        else:
            self.image_ = origin_frame
            
        if self.image_.shape[1] != 3:
            self.image_ = preprocessTarget(self.image_, 1, size)
        self.maps_generator_ = maps_generator

    def __getitem__(self, index):

        input_im = torch.cat([self.image_, self.maps_generator_(self.size_, time_stamp=index)[None]], dim=1)

        return input_im

    def __len__(self):
        return self.video_length_


from experiment.spatial_utils import plotter, prepareWriting
from utils.common_utils import optimize, write_video
import time
import json
import os

class Experiment:

    def __init__(self, config, optimize_parameters, batch_generator, net, loss, plotter=plotter):
        self.config_ = config
        self.optimize_parameters_ = optimize_parameters
        self.batch_generator_ = batch_generator
        self.net_ = net
        self.loss_ = loss
        self.plotter_ = plotter

    def run(self):
        i = 0

        def closure():
            nonlocal i
            X, Y = self.batch_generator_()
            Y_hat = self.net_(X)

            total_loss = self.loss_(Y, Y_hat)
            total_loss.backward()

            print('Iteration %05d    Loss %f' %
                  (i, total_loss.data[0]), '\r', end='')
            if self.config_["PLOT"] and i % self.config_["show_every"] == 0:
                self.plotter_(Y_hat)

            i += 1

            return total_loss

        optimize(self.config_["optimizer"], self.optimize_parameters_,
                 closure, self.config_["lr"], self.config_["num_iter"])

    def save_result(self):
        X = self.batch_generator_(mode='test')
        Y_hat = self.net_(X)
        video_predict = prepareWriting(Y_hat)
        file_name = time.strftime("%d_%b_%Y:%H:%M:%S", time.gmtime())

        path = 'experiment_results/{}'.format(file_name)
        self.path_ = path
        os.makedirs(path)

        write_video(path + "/predict.mp4", video_predict)

        X, Y = self.batch_generator_(mode='train')
        Y_hat = self.net_(X)
        video_fit = prepareWriting(Y_hat)
        write_video(path + "/fit.mp4", video_fit)

        video_target = prepareWriting(Y)
        write_video(path + "/target.mp4", video_target)

        with open(path + "/config.json", "w") as f:
            json.dump(self.config_, f)


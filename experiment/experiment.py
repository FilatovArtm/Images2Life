from utils.common_utils import optimize, write_video, plotter, prepareWriting
import time
import json
import os
import numpy as np

class Experiment2:

    def __init__(self, config, optimize_parameters, data_loader, net, loss, plotter=plotter):
        self.config_ = config
        self.optimize_parameters_ = optimize_parameters
        self.data_loader_ = data_loader
        self.net_ = net
        self.loss_ = loss
        self.plotter_ = plotter

    def run(self):
        iterator = iter(self.data_loader_)
        i = 0
        

        def closure():
            nonlocal iterator
            nonlocal i
            try:
                X, Y = next(iterator)
            except:
                iterator = iter(self.data_loader_)
                X, Y = next(iterator)
                
            X, Y = X.cuda(), Y.cuda()
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

    def predict_video(self, test_generator):
        result = []

        for i in range(len(test_generator)):
            X = test_generator[i].cuda()
            Y_hat = self.net_(X)
            result.append(prepareWriting(Y_hat))

        return np.concatenate(result)

    def save_result(self, test_generator, train_generator):
        video_predict = self.predict_video(test_generator)
        file_name = time.strftime("%d_%b_%Y:%H:%M:%S", time.gmtime())
        path = 'experiment_results/{}'.format(file_name)
        self.path_ = path
        os.makedirs(path)

        write_video(path + "/predict.mp4", video_predict)

        video_fit = self.predict_video(train_generator)
        write_video(path + "/fit.mp4", video_fit)

        #video_target = prepareWriting(self.dat.target_)
        #write_video(path + "/target.mp4", video_target)

        with open(path + "/config.json", "w") as f:
            json.dump(self.config_, f)



    

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
            X, Y = X.cuda(), Y.cuda()
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

    def predict_video(self, start, length):
        result = []

        for i in range(int(length / self.batch_generator_.batch_size_)):
            X = self.batch_generator_(mode='test', begin=start, n=i).cuda()
            Y_hat = self.net_(X)
            result.append(prepareWriting(Y_hat))

        return np.concatenate(result)

    def save_result(self):
        video_predict = self.predict_video(len(self.batch_generator_.target_), 64)
        file_name = time.strftime("%d_%b_%Y:%H:%M:%S", time.gmtime())
        path = 'experiment_results/{}'.format(file_name)
        self.path_ = path
        os.makedirs(path)

        write_video(path + "/predict.mp4", video_predict)

        video_fit = self.predict_video(0, len(self.batch_generator_.target_))
        write_video(path + "/fit.mp4", video_fit)

        video_target = prepareWriting(self.batch_generator_.target_)
        write_video(path + "/target.mp4", video_target)

        with open(path + "/config.json", "w") as f:
            json.dump(self.config_, f)


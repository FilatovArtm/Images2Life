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
                  (i, self.loss_.data[0]), '\r', end='')
            if self.config_["PLOT"] and i % self.config_["show_every"] == 0:
                self.plotter_(Y_hat)

            i += 1

            return total_loss

        optimize(self.config_["optimizer"], self.optimize_parameters_,
                 closure, self.config_["lr"], self.config_["num_iter"])

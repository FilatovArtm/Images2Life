config = {
    "PLOT": True,
    "optimizer": "adam",
    "lr": 1e-2,
    "num_iter": 2000,
    "show_every": 100,
    "maps_number": 4,
    "input_size": 128,
    "output_size": 128,
    "video_length": 64
}

skip_params = {'num_input_channels': input_depth,
               'num_channels_down': [8, 16, 24],
               'num_channels_up': [8, 16, 24],
               'num_channels_skip': [4, 4, 4]}

pregrid_params = {'num_input_channels': input_depth,
                  'num_output_channels': 2,
                  'num_channels_down': [8, 16, 24],
                  'num_channels_up': [8, 16, 24],
                  'num_channels_skip': [4, 4, 4]}

net = spatial.Net(input_depth=input_depth, pic_size=N, skip_args_main=skip_params,
                  skip_args_grid=pregrid_params).type(dtype)


video = generateSyntheticData()
target = preprocessTarget(video, config["video_length"], config["output_size"])
loss = SpatialLoss()
spatial_maps_generator = SpatialMapsGenerator(config["maps_number"])
batch_generator = BatchGenerator(target, spatial_maps_generator, config["maps_number"], config[
                                 "input_size"], config["input_size"])

parameters = [net.get_parameters()]
for var in spatial_maps_generator.spatial_variables.values():
    parameters.append(var)

experiment = Experiment(config, parameters, batch_generator, net, loss)
experiment.run()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.autograd import Variable

import numpy as np
import pickle
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self, num_hidden_units):
        super(Net, self).__init__()

        # 2 linear points:
        # input to hidden
        # hidden to output
        self.linear1 = nn.Linear(5, num_hidden_units)
        self.linear2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.linear3 = nn.Linear(num_hidden_units, 3)

        # self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimensionv
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def load_data(file_list):

    inputs = []
    outputs = []

    for file in file_list:

        data = np.load(file)

        inputs += data['inputs']
        outputs += data['outputs']

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return inputs, outputs


def main():
    import os
    from os import listdir, getcwd, chdir
    from os.path import isfile, join

    script_location = os.path.dirname(os.path.abspath(__file__))
    chdir(script_location + '/Data')

    data_files = [f for f in listdir(getcwd()) if isfile(join(getcwd(), f))]

    # print(data_files)

    import random
    random.shuffle(data_files)

    # print(data_files)

    training_files = data_files[:-1]
    testing_files = data_files[-1:]

    training_inputs, training_outputs = load_data(training_files)
    testing_inputs, testing_outputs = load_data(testing_files)

    # print(training_inputs.shape, training_outputs.shape)

    # print(testing_inputs.shape, testing_outputs.shape)

    num_hidden_units = 100
    network = Net(num_hidden_units).cuda()

    num_training_examples = len(training_inputs)
    batch_size = 1000
    num_epochs = 1000

    # exit()

    import math
    num_batches = math.ceil(num_training_examples / batch_size)

    for epoch in range(num_epochs):

        print("Epoch %d" % (epoch + 1))

        shuffling_indices = np.random.choice(num_training_examples, num_training_examples, replace=False)

        training_inputs = training_inputs[shuffling_indices]
        training_outputs = training_outputs[shuffling_indices]

        for batch in range(num_batches):

            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size, num_training_examples)

            input_batch = training_inputs[start_index:end_index]
            output_batch = training_outputs[start_index:end_index]

            x = Variable(torch.from_numpy(np.array(input_batch, dtype=np.float32))).cuda()
            y = Variable(torch.from_numpy(np.array(output_batch, dtype=np.float32))).cuda()

            network.optimizer.zero_grad()   # zero the gradient buffers
            output = network(x)
            loss = network.loss(output, y)
            loss.backward()
            network.optimizer.step()

        print("Train loss: %f" % loss.item())

        # test
        x = Variable(torch.from_numpy(np.array(testing_inputs, dtype=np.float32))).cuda()
        y = Variable(torch.from_numpy(np.array(testing_outputs, dtype=np.float32))).cuda()

        output = network(x)
        loss = network.loss(output, y)

        print("Test loss: %f" % loss.item())
    x = x.detach()
    y = y.detach()
    output = output.detach()

    plt.plot(x[:, -1].cpu(), y[:, 0].cpu())
    plt.plot(x[:, -1].cpu(), output[:, 0].cpu())
    print("Actual max", np.max(np.array(y.cpu()), axis=0)[0])
    print("Guessed max", np.max(np.array(output.cpu()), axis=0)[0])

    plt.legend(['actual', 'prediction'])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
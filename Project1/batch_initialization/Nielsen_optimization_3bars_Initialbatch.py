# overhead
import logging
import math
import random
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt
import torchvision.models as models

from ipywidgets import IntProgress
from IPython.display import display
from matplotlib import pyplot as plt, rc
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
rc('animation', html='jshtml')
from jupyterthemes import jtplot
jtplot.style(theme='grade3', context='notebook', ticks=True, grid=False)

logger = logging.getLogger(__name__)

# environment parameters
FRAME_TIME = 0.1  # time interval
OMEGA_RATE1 = 0.1  # max rotation rate
OMEGA_RATE2 = 0.1
OMEGA_RATE3 = 0.1
L1 = 1
L2 = 1
L3 = 1

class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    def forward(self, state, action):
        """
        action[0] = theta_dot_1
        action[1] = theta_dot_2
        action[2] = theta_dot_3

        state[0] = x
        state[1] = y
        state[2] = theta 1
        state[3] = theta 2
        state[4] = theta 3

        """
        # updata x and y data from theta
        state_3 = torch.zeros((10, 5))
        state_3[:, 0] = -1 * torch.cos(state[:, 2])*L1 -1 * torch.cos(state[:, 3])*L2 -1 * torch.cos(state[:, 4])*L3
        state_3[:, 1] = torch.sin(state[:, 2])*L1 + torch.sin(state[:, 3])*L2 + torch.sin(state[:, 4])*L3

        # limit theta dot 1
        state_1 = torch.zeros((10, 5))
        state_1[:, 2] = OMEGA_RATE1
        
        # using action variable to control theta dot 1
        delta_state_1 = FRAME_TIME * torch.mul(state_1, action[:, 0].reshape(-1, 1))
        
        # limit theta dot 2
        state_2 = torch.zeros((10, 5))
        state_2[:, 3] = OMEGA_RATE2
        
        # using action variable to control theta dot 2
        delta_state_2 = FRAME_TIME * torch.mul(state_2, action[:, 1].reshape(-1, 1))
        
        # limit theta dot 3
        state_4 = torch.zeros((10, 5))
        state_4[:, 4] = OMEGA_RATE3
        
        # using action variable to control theta dot 3
        delta_state_4 = FRAME_TIME * torch.mul(state_4, action[:, 2].reshape(-1, 1))


        # Update state
        step_mat = torch.tensor([[0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 1., 0., 0.],
                                 [0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 1.]])


        state = torch.matmul(step_mat, state.T)
        state = state.T

        state = state + delta_state_1 + delta_state_2 + state_3 + delta_state_4

        return state


class Controller(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden:
        """
        super(Controller, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            nn.Sigmoid())

    def forward(self, state):
        action = self.network(state)
        action = (action - torch.tensor([0.5, 0.5, 0.5])) * 2  # bound theta_dot action variable range (-1 to 1)* omegarate
        return action


class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.theta_trajectory = torch.empty((1, 0))
        self.u_trajectory = torch.empty((1, 0))

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller(state)
            state = self.dynamics(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        # state = [[-3.,0.,0.,0.,0.]]  # Making a random batch initialization
        state = abs( 2. * np.pi * torch.rand((10, 5)))
        state[:, 0] = -1 * torch.cos(state[:, 2])*L1 -1 * torch.cos(state[:, 3])*L2 -1 * torch.cos(state[:, 4])*L3
        state[:, 1] =  torch.sin(state[:, 2])*L1 + torch.sin(state[:, 3])*L2 + torch.sin(state[:, 4])*L3

        return torch.tensor(state, requires_grad=False).float()

    def error(self, state):
        # reduce error at point (0.5, 1.5)
        return torch.mean((state[:, 1] - 1.5) ** 2) + torch.mean((state[:, 0] - 0.5) ** 2)


class Optimize:

    # create properties of the class (simulation, parameters, optimizer, lost_list). Where to receive input of objects

    def __init__(self, simulation):
        self.simulation = simulation  # define the objective function
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)  # define the opmization algorithm
        self.loss_list = []

    # Define loss calculation method for objective function

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)  # calculate the loss of objective function
            self.optimizer.zero_grad()
            loss.backward()  # calculate the gradient
            return loss

        self.optimizer.step(closure)
        return closure()

    # Define training method for the model

    def train(self, epochs):
        # self.optimizer = epoch
        l = np.zeros(epochs)
        for epoch in range(epochs):
            self.epoch = epoch
            loss = self.step()  # use step function to train the model
            self.loss_list.append(loss)  # add loss to the loss_list
            print('[%d] loss: %.3f' % (epoch + 1, loss))

            l[epoch] = loss
            self.visualize()

        plt.plot(list(range(epochs)), l)

        plt.title('Objective Function Convergence Curve')
        plt.xlabel('Training Iteration')
        plt.ylabel('Error')
        plt.show()
        self.animation(epochs)

    # Define result visualization method

    def visualize(self):

        if self.epoch == 49:

            data = np.array(
                [[self.simulation.state_trajectory[i][N].detach().numpy() for i in range(self.simulation.T)] for N in
                 range(10)])
            action_data = np.array(
                [[self.simulation.action_trajectory[i][N].detach().numpy() for i in range(self.simulation.T)] for N in
                 range(10)])

            fig, ax = plt.subplots(1, 3, tight_layout=1, figsize=(15, 5))
            fig2, ax1 = plt.subplots()
            circle1 = plt.Circle((0.5, 1.5), 0.1)
            # only print last training data
            for i in range(10):
                x = data[i, :, 0]
                y = data[i,:, 1]
                theta1 = data[i, :, 2]
                theta2 = data[i, :, 3]
                theta3 = data[i, :, 4]

                thetadot1 = action_data[i, :, 0]
                thetadot2 = action_data[i, :, 1]
                thetadot3 = action_data[i, :, 2]

                frame = range(self.simulation.T)

                ax1.plot(x, y)
                ax1.add_patch( circle1 )
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.set_aspect(1)
                ax1.set(title=f'Tip Location after {self.epoch} Trainings')

                ax[0].plot(frame, theta1)
                ax[0].set_xlabel("Time")
                ax[0].set_ylabel("Theta")
                ax[0].set(title=f'Theta 1 after {self.epoch} Trainings')

                ax[1].plot(frame, theta2)
                ax[1].set_xlabel("Time")
                ax[1].set_ylabel("Theta")
                ax[1].set(title=f'Theta 2 after {self.epoch} Trainings')

                ax[2].plot(frame, theta3)
                ax[2].set_xlabel("Time")
                ax[2].set_ylabel("Theta")
                ax[2].set(title=f'Theta 3 after {self.epoch} Trainings')
                #
                # ax2[0].plot(frame, thetadot1)
                # ax2[0].set_xlabel("Time")
                # ax2[0].set_ylabel("Theta dot")
                # ax2[0].set(title=f'Theta dot 1 after {self.epoch} Trainings')
                #
                # ax2[1].plot(frame, thetadot2)
                # ax2[1].set_xlabel("Time")
                # ax2[1].set_ylabel("Theta dot")
                # ax2[1].set(title=f'Theta dot 2 after {self.epoch} Trainings')
                #
                # ax2[2].plot(frame, thetadot3)
                # ax2[2].set_xlabel("Time")
                # ax2[2].set_ylabel("Theta dot")
                # ax2[2].set(title=f'Theta dot 3 after {self.epoch} Trainings')
            plt.show()

    def animation(self, epochs):
        print("Generating Animation")
        steps = 500 + 1
        final_time_step = round(1 / steps, 2)
        f = IntProgress(min=0, max=steps)
        display(f)

        data = np.array(
            [[self.simulation.state_trajectory[i][N].detach().numpy() for i in range(self.simulation.T)] for N in
             range(10)])

        fig = plt.figure(figsize=(5, 8), constrained_layout=False)
        ax1 = fig.add_subplot(111)
        plt.axhline(y=0., color='b', linestyle='--', lw=0.8)


        # sample 1
        lnt_1, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_1, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_1, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_1, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3
        # sample 2
        lnt_2, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_2, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_2, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_2, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 3
        lnt_3, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_3, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_3, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_3, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 4
        lnt_4, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_4, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_4, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_4, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 5
        lnt_5, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_5, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_5, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_5, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 6
        lnt_6, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_6, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_6, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_6, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 7
        lnt_7, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_7, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_7, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_7, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 8
        lnt_8, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_8, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_8, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_8, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 9
        lnt_9, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_9, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_9, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_9, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        # sample 10
        lnt_10, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1_10, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2_10, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3_10, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 3

        plt.tight_layout()

        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-2, 5)
        ax1.set_aspect(1)  # aspect of the axis scaling, i.e. the ratio of y-unit to x-unit

        def update(i):
            # sample 1
            theta1 = data[0, i, 2]
            theta2 = data[0, i, 3]
            theta3 = data[0, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_1.set_data(Bar1_x, Bar1_y)
            ln2_1.set_data(Bar2_x, Bar2_y)
            ln3_1.set_data(Bar3_x, Bar3_y)

            lnt_1.set_data(data[0, :i, 0], data[0, :i, 1])

            # sample 2
            theta1 = data[1, i, 2]
            theta2 = data[1, i, 3]
            theta3 = data[1, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_2.set_data(Bar1_x, Bar1_y)
            ln2_2.set_data(Bar2_x, Bar2_y)
            ln3_2.set_data(Bar3_x, Bar3_y)

            lnt_2.set_data(data[1, :i, 0], data[1, :i, 1])

            # sample 3
            theta1 = data[2, i, 2]
            theta2 = data[2, i, 3]
            theta3 = data[2, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_3.set_data(Bar1_x, Bar1_y)
            ln2_3.set_data(Bar2_x, Bar2_y)
            ln3_3.set_data(Bar3_x, Bar3_y)

            lnt_3.set_data(data[2, :i, 0], data[2, :i, 1])

            # sample 4
            theta1 = data[3, i, 2]
            theta2 = data[3, i, 3]
            theta3 = data[3, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_4.set_data(Bar1_x, Bar1_y)
            ln2_4.set_data(Bar2_x, Bar2_y)
            ln3_4.set_data(Bar3_x, Bar3_y)

            lnt_4.set_data(data[3, :i, 0], data[3, :i, 1])

            # sample 5
            theta1 = data[4, i, 2]
            theta2 = data[4, i, 3]
            theta3 = data[4, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_5.set_data(Bar1_x, Bar1_y)
            ln2_5.set_data(Bar2_x, Bar2_y)
            ln3_5.set_data(Bar3_x, Bar3_y)

            lnt_5.set_data(data[4, :i, 0], data[4, :i, 1])

            # sample 6
            theta1 = data[5, i, 2]
            theta2 = data[5, i, 3]
            theta3 = data[5, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_6.set_data(Bar1_x, Bar1_y)
            ln2_6.set_data(Bar2_x, Bar2_y)
            ln3_6.set_data(Bar3_x, Bar3_y)

            lnt_6.set_data(data[5, :i, 0], data[5, :i, 1])

            # sample 7
            theta1 = data[6, i, 2]
            theta2 = data[6, i, 3]
            theta3 = data[6, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_7.set_data(Bar1_x, Bar1_y)
            ln2_7.set_data(Bar2_x, Bar2_y)
            ln3_7.set_data(Bar3_x, Bar3_y)

            lnt_7.set_data(data[6, :i, 0], data[6, :i, 1])

            # sample 8
            theta1 = data[7, i, 2]
            theta2 = data[7, i, 3]
            theta3 = data[7, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_8.set_data(Bar1_x, Bar1_y)
            ln2_8.set_data(Bar2_x, Bar2_y)
            ln3_8.set_data(Bar3_x, Bar3_y)

            lnt_8.set_data(data[7, :i, 0], data[7, :i, 1])

            # sample 9
            theta1 = data[8, i, 2]
            theta2 = data[8, i, 3]
            theta3 = data[8, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_9.set_data(Bar1_x, Bar1_y)
            ln2_9.set_data(Bar2_x, Bar2_y)
            ln3_9.set_data(Bar3_x, Bar3_y)

            lnt_9.set_data(data[8, :i, 0], data[8, :i, 1])

            # sample 10
            theta1 = data[9, i, 2]
            theta2 = data[9, i, 3]
            theta3 = data[9, i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1_10.set_data(Bar1_x, Bar1_y)
            ln2_10.set_data(Bar2_x, Bar2_y)
            ln3_10.set_data(Bar3_x, Bar3_y)

            lnt_10.set_data(data[9, :i, 0], data[9, :i, 1])


            f.value += 1

        anim = FuncAnimation(fig, update, np.arange(0, steps - 1, 25), interval=0.000001)

        plt.show()

        writer = PillowWriter(fps=20)
        anim.save("batch_animation.gif", writer=writer)


T = 500  # number of time steps of the simulation
dim_input = 5  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 3  # action space dimensions
d = Dynamics()
c = Controller(dim_input, dim_hidden, dim_output)
s = Simulation(c, d, T)
o = Optimize(s)
o.train(50)  # training with number of epochs (gradient descent steps)

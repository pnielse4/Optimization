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
        state_3 = torch.zeros((1, 5))
        state_3[0, 0] = -1 * torch.cos(state[0, 2])*L1 -1 * torch.cos(state[0, 3])*L2 -1 * torch.cos(state[0, 4])*L3
        state_3[0, 1] = torch.sin(state[0, 2])*L1 + torch.sin(state[0, 3])*L2 + torch.sin(state[0, 4])*L3

        # limit theta dot 1
        state_1 = torch.zeros((1, 5))
        state_1[0, 2] = OMEGA_RATE1

        # using action variable to control theta dot 1
        delta_state_1 = FRAME_TIME * torch.mul(state_1, action[0, 0].reshape(-1, 1))

        # limit theta dot 2
        state_2 = torch.zeros((1, 5))
        state_2[0, 3] = OMEGA_RATE2

        # using action variable to control theta dot 2
        delta_state_2 = FRAME_TIME * torch.mul(state_2, action[0, 1].reshape(-1, 1))

        # limit theta dot 3
        state_4 = torch.zeros((1, 5))
        state_4[0, 4] = OMEGA_RATE3

        # using action variable to control theta dot 3
        delta_state_4 = FRAME_TIME * torch.mul(state_4, action[0, 2].reshape(-1, 1))


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
        state = [[-3.,0.,0.,0.,0.]] # initialize
        return torch.tensor(state, requires_grad=False).float()

    def error(self, state):
        # reduce error at point (0.5, 1.5)
        return torch.mean((state[0,1] - 1.5) ** 2) + torch.mean((state[0,0] - 0.5) ** 2)


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
        # only print last training data
        if self.epoch == 49:
            data = np.array([self.simulation.state_trajectory[i][0].detach().numpy() for i in range(self.simulation.T)])
            x = data[:, 0]
            y = data[:, 1]
            theta1 = data[:, 2]
            theta2 = data[:, 3]
            theta3 = data[:, 4]
            frame = range(self.simulation.T)
            circle1 = plt.Circle((0.5, 1.5), 0.1)

            fig, ax = plt.subplots(1, 2, tight_layout=1, figsize=(15, 5))

            ax[0].plot(x, y, c='b')
            ax[0].add_patch(circle1)
            ax[0].set_xlabel("X")
            ax[0].set_ylabel("Y")
            ax[0].set(title=f'Tip Location after {self.epoch + 1} Trainings')

            ax[1].plot(frame, theta1, c='c', label="theta 1")
            ax[1].plot(frame, theta2, c='r', label="theta 2")
            ax[1].plot(frame, theta3, c='orange', label="theta 3")
            ax[1].set_xlabel("Time")
            ax[1].set_ylabel("Theta")
            ax[1].legend(frameon=0)
            ax[1].set(title=f'Theta after {self.epoch + 1} Trainings')
            plt.show()



    def animation(self, epochs):

        print("Generating Animation")
        steps = 500 + 1
        f = IntProgress(min=0, max=steps)
        display(f)

        data = np.array([self.simulation.state_trajectory[i][0].detach().numpy() for i in range(self.simulation.T)])
        action_data = np.array(
            [self.simulation.action_trajectory[i][0].detach().numpy() for i in range(self.simulation.T)])

        x_t = data
        u_t = action_data
        print(x_t.shape, u_t.shape)

        fig = plt.figure(figsize=(5, 8), constrained_layout=False)
        ax1 = fig.add_subplot(111)
        plt.axhline(y=0., color='b', linestyle='--', lw=0.8)


        ln6, = ax1.plot([], [], '--', linewidth=2, color='orange')  # trajectory line
        ln1, = ax1.plot([], [], linewidth=5, color='lightblue')  # Bar 1
        ln2, = ax1.plot([], [], linewidth=5, color='tomato')  # Bar 2
        ln3, = ax1.plot([], [], linewidth=5, color='orange')  # Bar 2

        plt.tight_layout()

        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-2, 5)
        ax1.set_aspect(1)  # aspect of the axis scaling, i.e. the ratio of y-unit to x-unit

        def update(i):
            theta1 = x_t[i, 2]
            theta2 = x_t[i, 3]
            theta3 = x_t[i, 4]

            Bar1_x = [0. , - np.cos(theta1)]
            Bar1_y = [0. , np.sin(theta1)]
            Bar2_x = [- np.cos(theta1) , - np.cos(theta1) - np.cos(theta2)]
            Bar2_y = [np.sin(theta1) , np.sin(theta1) + np.sin(theta2)]
            Bar3_x = [- np.cos(theta1) - np.cos(theta2), - np.cos(theta1) - np.cos(theta2) - np.cos(theta3)]
            Bar3_y = [ np.sin(theta1) + np.sin(theta2), np.sin(theta1) + np.sin(theta2) + np.sin(theta3)]


            ln1.set_data(Bar1_x, Bar1_y)
            ln2.set_data(Bar2_x, Bar2_y)
            ln3.set_data(Bar3_x, Bar3_y)

            ln6.set_data(x_t[:i, 0], x_t[:i, 1])

            f.value += 1

        anim = FuncAnimation(fig, update, np.arange(0, steps - 1, 25), interval=0.000001)

        plt.show()

        writer = PillowWriter(fps=20)
        anim.save("3bar_arm_animation.gif", writer=writer)


T = 500  # number of time steps of the simulation
dim_input = 5  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 3  # action space dimensions
d = Dynamics()
c = Controller(dim_input, dim_hidden, dim_output)
s = Simulation(c, d, T)
o = Optimize(s)
o.train(50)  # training with number of epochs (gradient descent steps)

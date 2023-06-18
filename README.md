# Robot arm

This code is a Python script that uses the NumPy and Matplotlib libraries to visualize a simple animation of a robotic arm. The script allows the user to interact with the animation by clicking on the plot window to change the target point for the arm to reach. The arm adjusts its angles (theta values) to move towards the target point.

Dependencies
NumPy: A library for numerical computing with Python.
Matplotlib: A plotting library for creating static, animated, and interactive visualizations in Python.
Make sure you have these libraries installed before running the script.

Usage
Import the necessary libraries:
python
Copy code
import numpy as np
import sys
import matplotlib
Set the appropriate backend for Matplotlib based on the operating system:
python
Copy code
if sys.platform == 'darwin':
    matplotlib.use("MacOSX")  # for macOS
else:
    matplotlib.use("TkAgg")  # for UNIX/Windows
Import the required modules from Matplotlib and configure the plot:
python
Copy code
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 7)  # size of the window
plt.ion()  # turn on interactive mode
plt.style.use('dark_background')  # set the plot style
Define the initial target point and anchor point for the arm:
python
Copy code
target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])
Set up event handling functions for button press, key press, and window close events:
python
Copy code
def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False  # quit the application

def on_close(event):
    global is_running
    is_running = False
Create the figure and connect the event handling functions to the corresponding events:
python
Copy code
fig, _ = plt.subplots()
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)
Define helper functions for rotation matrices:
python
Copy code
def rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c, -s],
        [s, c]
    ])
    return R

def d_rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [-s, -c],
        [c, -s]
    ])
    return R
Enter the main animation loop:
python
Copy code
while is_running:
    plt.clf()  # clear the plot

    # Calculate joint positions based on current theta values
    segment = np.array([0.0, 1.0]) * length_joint
    joints = []
    # Calculate rotation matrices
    R1 = rotation(theta_1)
    R2 = rotation(theta_2)
    R3 = rotation(theta_3)

    # Calculate derivative of rotation matrices
    dR1 = d_rotation(theta_1)
    dR2 = d_rotation(theta_2)
    dR3 = d_rotation(theta_3)

    joints.append(anchor_point)

    point_1 = R1 @ segment
    joints.append(point_1)

    point_2 = point_1 + R1 @ (R2 @ segment)
    joints.append(point_2)

    point_3 = point_2 + R2 @ (R3 @ segment)
    joints.append(point_3)

    np_joints = np.array(joints)

    # Calculate loss and gradients
    distance = np.sqrt(np.sum((point_3 - target_point) ** 2))
    loss = np.sum((point_3 - target_point) ** 2) + 0.1 * distance

    d_loss = -2 * (target_point - point_3)
    d_theta_1 = d_loss * (R2 @ dR1 @ segment)
    d_theta_2 = d_loss * (R1 @ dR2 @ (R2 @ segment))
    d_theta_3 = d_loss * (R1 @ R2 @ dR3 @ segment)

    # Apply additional error terms
    additional_error_3 = np.sum(np.maximum(joints[3][1] - joints[1][1], 0) ** 2)
    additional_error_2 = np.sum(np.maximum(joints[2][1] - joints[1][1], 0) ** 2)
    additional_error_1 = np.sum(np.maximum(joints[1][1], 0) ** 2)

    d_theta_3 += 0.1 * additional_error_3 * (R2 @ R1 @ dR3 @ segment)
    d_theta_2 += 0.1 * additional_error_2 * (R1 @ dR2 @ segment)
    d_theta_1 += 0.1 * additional_error_1 * (dR1 @ segment)

    # Update theta values using gradient descent
    alpha = 1e-2
    theta_1 -= np.sum(d_theta_1 * alpha)
    theta_2 -= np.sum(d_theta_2 * alpha)
    theta_3 -= np.sum(d_theta_3 * alpha)

    # Update the plot with current state
    plt.title(
        f'theta_1: {round(np.rad2deg(theta_1))} '
        f'theta_2: {round(np.rad2deg(theta_2))} '
        f'theta_3: {round(np.rad2deg(theta_3))} '
        f'loss: {loss} '
    )

    if len(np_joints):
        plt.plot(np_joints[:, 0], np_joints[:, 1])
    plt.scatter(target_point[0], target_point[1], s=50, c='r')

    plt.xlim(-5, 5)
    plt.ylim(0, 10)

    # Draw the plot and pause for a short time
    plt.draw()
    plt.pause(1e-3)
Make sure to adjust any parameters or settings to suit your needs. You can modify the code and experiment with different configurations to understand how the robotic arm behaves.

import numpy as np
import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("MacOSX") # for mac
else:
    matplotlib.use("TkAgg") # for unix/windows

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 7) # size of window
plt.ion()
plt.style.use('dark_background')

target_point = np.array([-3.0, 0])
anchor_point = np.array([0, 0])

is_running = True
def button_press_event(event):
    global target_point
    target_point = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app

def on_close(event):
    global is_running
    is_running = False

fig, _ = plt.subplots()
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length_joint = 2.0
theta_1 = np.deg2rad(-10)
theta_2 = np.deg2rad(-10)
theta_3 = np.deg2rad(-10)

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


while is_running:
    plt.clf()

    segment = np.array([0.0, 1.0]) * length_joint
    joints = []

    R1 = rotation(theta_1)
    R2 = rotation(theta_2)
    R3 = rotation(theta_3)

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

    distance = np.sqrt(np.sum((point_3 - target_point) ** 2))
    loss = np.sum((point_3 - target_point) ** 2) + 0.1 * distance

    d_loss = -2 * (target_point - point_3)
    d_theta_1 = d_loss * (R2 @ dR1 @ segment)
    d_theta_2 = d_loss * (R1 @ dR2 @ (R2 @ segment))
    d_theta_3 = d_loss * (R1 @ R2 @ dR3 @ segment)

    additional_error_3 = np.sum(np.maximum(joints[3][1] - joints[1][1], 0) ** 2)
    additional_error_2 = np.sum(np.maximum(joints[2][1] - joints[1][1], 0) ** 2)
    additional_error_1 = np.sum(np.maximum(joints[1][1], 0) ** 2)

    d_theta_3 += 0.1 * additional_error_3 * (R2 @ R1 @ dR3 @ segment)
    d_theta_2 += 0.1 * additional_error_2 * (R1 @ dR2 @ segment)
    d_theta_1 += 0.1 * additional_error_1 * (dR1 @ segment)

    alpha = 1e-2
    theta_1 -= np.sum(d_theta_1 * alpha)
    theta_2 -= np.sum(d_theta_2 * alpha)
    theta_3 -= np.sum(d_theta_3 * alpha)

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
    plt.draw()
    plt.pause(1e-3)
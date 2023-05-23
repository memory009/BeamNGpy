"""
.. module:: lidar_tour
    :platform: Windows
    :synopsis: Example starting in west_coast_usa with a vehicle that has a
               Lidar attached and drives around the environment using the
               builtin AI. Lidar data is displayed using the OpenGL-based
               Lidar visualiser.
.. moduleauthor:: Marc Müller <mmueller@beamng.gmbh>
"""

import numpy as np
import pickle as pkl

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Lidar
from beamngpy.sensors.lidar import MAX_LIDAR_POINTS, LidarVisualiser

SIZE = 1024


def lidar_resize(width, height):
    if height == 0:
        height = 1

    glViewport(0, 0, width, height)


def open_window(width, height):
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
    glutInitWindowSize(width, height)
    window = glutCreateWindow(b'Lidar Tour')
    lidar_resize(width, height)
    return window

def rotate(points, n):
    # Step 1
    n = n / np.linalg.norm(n)
    # Step 2
    theta = np.arccos(n[0])
    # phi = np.arctan2(n[1], n[2])
    # print(phi)
    # Step 3
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    # R = np.dot(R, np.array([[1, 0, 0],
    #                         [0, np.cos(phi), -np.sin(phi)],
    #                         [0, np.sin(phi), np.cos(phi)]]))
    rotated_points = np.dot(points, R.T)
    return rotated_points

# def rotate(points, u):
#     # Step 1
#     u = u / np.linalg.norm(u)
#     print(u)
#     # Step 2
#     theta = np.arccos(u[0])
#     # Step 3
#     R = np.array([[np.cos(theta), -np.sin(theta), 0],
#                   [np.sin(theta), np.cos(theta), 0],
#                   [0, 0, 1]])
#     rotated_points = np.dot(points, R.T)
#     return rotated_points

def main():
    set_up_simple_logging()

    beamng = BeamNGpy('localhost', 64256)
    bng = beamng.open(launch=True)
    scenario = Scenario('autobahnvawriss', 'lidar_tour', description='Tour through the west coast gathering Lidar data')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='LIDAR')

    scenario.add_vehicle(vehicle, pos=(-17344.418, -7.627, 246.561), rot_quat=(0, 0, 0, 1))
    scenario.make(bng)

    try:
        bng.scenario.load(scenario)

        window = open_window(SIZE, SIZE)
        lidar_vis = LidarVisualiser(MAX_LIDAR_POINTS)
        lidar_vis.open(SIZE, SIZE)

        bng.settings.set_deterministic(60)
        # bng.ui.hide_hud()
        bng.scenario.start()

        # Send data via shared memory.
        lidar = Lidar('lidar', bng, vehicle, requested_update_time=0.01, is_using_shared_memory=True)
        # lidar = Lidar('lidar', bng, vehicle, requested_update_time=0.01, is_using_shared_memory=False)   # Send data through lua socket instead.

        bng.control.pause()
        vehicle.ai.set_mode('disabled')

        def update():
            vehicle.sensors.poll()
            points = lidar.poll()['pointCloud']
            dir_lidar = np.array(lidar.get_direction())
            pos_lidar = np.array(lidar.get_position())

            points_np = np.array(points).reshape(-1, 3)
            points_np = points_np[points_np != [0, 0, 0]].reshape(-1, 3)
            points_np = points_np - pos_lidar[np.newaxis, :]
            points_np = rotate(points_np, dir_lidar)
            count = points_np.shape[0]
            _ = np.zeros(shape = (count,4))
            _[:, 0:3] = points_np
            points_np = _
            # print(f"lidar pos:{pos_lidar}")
            print(f"pc count: {count}")
            # print(f"pc: {points_np}")
            # print(f"lidar dir: {dir_lidar}")
            points_np.astype(np.float32).tofile("./test_pc.bin")

            bng.step(3, wait=False)

            lidar_vis.update_points(points, vehicle.state)
            glutPostRedisplay()

        glutReshapeFunc(lidar_resize)
        glutIdleFunc(update)
        glutMainLoop()
    finally:
        bng.close()


if __name__ == '__main__':
    main()

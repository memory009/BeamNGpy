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
    # step 1 normalization归一化 
    n = n / np.linalg.norm(n)

    # Step 2 计算向量n与x轴正半轴之间的夹角(弧度制)，并将该夹角保存在变量theta中
    theta = np.arccos(n[0])
    # phi = np.arctan2(n[1], n[2])
    # print(phi)

    # Step 3 定义一个旋转矩阵R，第一行对应x轴旋转，第二行对应y轴旋转，第三行对应z轴旋转，且旋转角度为theta
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    # R = np.dot(R, np.array([[1, 0, 0],
    #                         [0, np.cos(phi), -np.sin(phi)],
    #                         [0, np.sin(phi), np.cos(phi)]]))
    rotated_points = np.dot(points, R.T)
    return rotated_points

def main():
    set_up_simple_logging()

    beamng = BeamNGpy('localhost', 64256, 'D:\BeamNG.tech.v0.27.2.0')
    bng = beamng.open(launch=True)
    scenario = Scenario('west_coast_usa', 'lidar_tour', description='Tour through the west coast gathering Lidar data')
    # vehicle = Vehicle('ego_vehicle', model='etk800', license='LIDAR')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='LIDAR')

    scenario.add_vehicle(vehicle, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)

    try:
        bng.scenario.load(scenario)

        window = open_window(SIZE, SIZE)
        lidar_vis = LidarVisualiser(MAX_LIDAR_POINTS)
        lidar_vis.open(SIZE, SIZE)
        

        bng.settings.set_deterministic(60)
        # 是否隐藏hub
        # bng.ui.hide_hud()
        bng.scenario.start()

        # Send data via shared memory.
        lidar = Lidar('lidar', bng, vehicle, requested_update_time=0.01, is_using_shared_memory=True)
        # lidar = Lidar('lidar', bng, vehicle, requested_update_time=0.01, is_using_shared_memory=False)   # Send data through lua socket instead.

        lidar.pc_index = 0
        
        bng.control.pause()
        # vehicle.ai.set_mode('span')
        # 切换到手动模式
        vehicle.ai.set_mode('disabled')
        

        def update():
            vehicle.sensors.poll()
            points = lidar.poll()['pointCloud']
            bng.step(3, wait=False)

            lidar_vis.update_points(points, vehicle.state)
            # The direction of the lidar
            dir_lidar = np.array(lidar.get_direction())
            # The position of the lidar
            pos_lidar = np.array(lidar.get_position())

            points_np = np.array(points).reshape(-1, 3)
            points_np = points_np[points_np !=[0, 0, 0]].reshape(-1, 3)
            points_np = points_np - pos_lidar[np.newaxis, : ]
            
            # 加入旋转矩阵，调整相机的朝向
            points_np = rotate(points_np, dir_lidar)

            # 将nx3的数组以nx4的形式输出
            count = points_np.shape[0]
            points_np_kitti = np.zeros((count, 4))
            points_np_kitti[:, :3] = points_np

            # 批量保存
            lidar.pc_index = lidar.pc_index + 1

            print(f"lidar pos :{pos_lidar}")
            print(f"coun: {count}")
            # print(f"point clouds:{points_np}")
            print(f"pointclouds_kitti: {points_np_kitti}")
            
            # save pointclouds data as .bin
            points_np_kitti.astype(np.float32).tofile(f'./output/output{lidar.pc_index}.bin')
            glutPostRedisplay()

        glutReshapeFunc(lidar_resize)
        glutIdleFunc(update)
        glutMainLoop()
    finally:
        bng.close()


if __name__ == '__main__':
    main()
 
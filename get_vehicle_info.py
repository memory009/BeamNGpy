from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

set_up_simple_logging()
# 创建一个BeamNGpy对象，并连接到BeamNG.drive实例
beamng = BeamNGpy('localhost', 64256, 'D:\BeamNG.tech.v0.27.2.0')
bng = beamng.open(launch=True)
scenario = Scenario('west_coast_usa', 'vehicle_info_test')
vehicle = Vehicle('test_car', model='etk800')

# 将车辆添加到场景中，并启动模拟
scenario.add_vehicle(vehicle, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
scenario.make(bng)

try:
    beamng.load_scenario(scenario)
    beamng.start_scenario()

    # 查询车辆模型信息
    # 没get_model_info()这个函数
    model_info = vehicle.get_model_info()
    print(model_info)

finally:
    bng.close()

import time
import argparse
from etils import epath
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

class LidarVisualizer:
    def __init__(self, mj_model):
        self.site_name = "lidar_site"

        self.rays_theta, self.rays_phi = scan_gen.generate_airy96()

        # 优化内存布局
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32) #[::4]
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32) #[::4]

        geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
        # geomgroup[1] = 0  # 排除group 1中的几何体
        self.lidar = MjLidarWrapper(mj_model, site_name=self.site_name, backend="gpu", args={'bodyexclude': mj_model.body("mocap_body").id, "geomgroup":geomgroup})

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与ROS2集成')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    args = parser.parse_args()

    print("=" * 60)
    print("MuJoCo LiDAR可视化")
    print("=" * 60)
    print(f"配置：")
    print(f"- LiDAR型号: Robosense airy-96")
    print(f"- 循环频率: 10 Hz")

    print("===================== LiDAR 模拟使用说明 =====================")
    print("1. 双击选中空中的绿色方块（模拟激光雷达的位置），")
    print("   按下ctrl，点击鼠标左键，拖动鼠标可以旋转绿色方块，")
    print("   按下ctrl和鼠标右键，拖动鼠标可以平移绿色方块")
    print("2. 按 Tab 键切换左侧 UI 的可视化界面；")
    print("   按 Shift+Tab 键切换右侧 UI 的可视化界面。")
    print("=" * 60)

    mjcf_file = epath.Path(__file__).parent.parent / "models" / "mjcf" / "mocap_env.xml"
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file.as_posix())
    mj_data = mujoco.MjData(mj_model)

    # 创建节点并运行
    node = LidarVisualizer(mj_model)

    # 创建定时器
    step_cnt = 0
    render_fps = 60
    step_gap = render_fps // 12

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 设置视图模式为site
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
        # viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value

        viewer.user_scn.ngeom = node.rays_theta.shape[0]
        for i in range(viewer.user_scn.ngeom):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.8])
            )
        print("Starting simulation...")
        print("Number of rays:", node.rays_theta.shape[0])

        cmap = plt.get_cmap('hsv')  # 或使用 'jet', 'viridis', 'plasma' 等

        while viewer.is_running():

            # 更新模拟
            for _ in range(int(1. / (render_fps * mj_model.opt.timestep))):
                mujoco.mj_step(mj_model, mj_data)
            step_cnt += 1
            viewer.sync()

            if step_cnt % step_gap == 0:
                
                start_time = time.time()
                node.lidar.trace_rays(mj_data, node.rays_theta, node.rays_phi)
                end_time = time.time()

                local_points = node.lidar.get_hit_points()
                world_points = local_points @ node.lidar.sensor_rotation.T + node.lidar.sensor_position

                # 根据高度设置颜色
                z_values = world_points[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                if z_max > z_min:
                    # 归一化高度值到 [0, 1]
                    z_norm = (z_values - z_min) / (z_max - z_min)
                else:
                    z_norm = np.zeros_like(z_values)
                
                # 使用 matplotlib 颜色映射
                colors = cmap(z_norm)  # 返回 RGBA 值，shape: (N, 4)
                
                for i in range(viewer.user_scn.ngeom):
                    viewer.user_scn.geoms[i].pos[:] = world_points[i]
                    viewer.user_scn.geoms[i].rgba[:] = colors[i]

                # 打印性能信息和当前位置
                if args.verbose:
                    # 格式化欧拉角为度数
                    lidar_orientation = Rotation.from_matrix(node.lidar.sensor_rotation).as_quat()
                    lidar_position = node.lidar.sensor_position
                    euler_deg = Rotation.from_quat(lidar_orientation).as_euler('xyz', degrees=True)
                    node.get_logger().info(f"位置: [{lidar_position[0]:.2f}, {lidar_position[1]:.2f}, {lidar_position[2]:.2f}], "
                        f"欧拉角: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°], "
                        f"耗时: {(end_time - start_time)*1000:.2f} ms")


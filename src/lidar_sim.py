import os
import time
import argparse
import threading
import traceback

import mujoco
import mujoco.viewer
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

def publish_point_cloud(publisher, points, frame_id, stamp):
    """将点云数据发布为ROS PointCloud2消息"""

    # 定义点云字段
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]

    # 添加强度值
    if len(points.shape) == 2:
        # 如果是(N, 3)形状，转换为(3, N)以便处理
        points_transposed = points.T if points.shape[1] == 3 else points

        if points_transposed.shape[0] == 3:
            # 添加强度通道
            points_with_intensity = np.vstack([
                points_transposed, 
                np.ones(points_transposed.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points_transposed
    else:
        # 如果点云已经是(3, N)形状
        if points.shape[0] == 3:
            points_with_intensity = np.vstack([
                points, 
                np.ones(points.shape[1], dtype=np.float32)
            ])
        else:
            points_with_intensity = points

    # 创建ROS2 PointCloud2消息
    pc_msg = PointCloud2()
    pc_msg.header.frame_id = frame_id
    pc_msg.header.stamp = stamp
    pc_msg.fields = fields
    pc_msg.is_bigendian = False
    pc_msg.point_step = 16  # 4 个 float32 (x,y,z,intensity)
    pc_msg.row_step = pc_msg.point_step * points_with_intensity.shape[1]
    pc_msg.height = 1
    pc_msg.width = points_with_intensity.shape[1]
    pc_msg.is_dense = True

    # 转置回(N, 4)格式并转换为字节数组
    pc_msg.data = np.transpose(points_with_intensity).astype(np.float32).tobytes()

    publisher.publish(pc_msg)

def broadcast_tf(broadcaster, parent_frame, child_frame, translation, rotation, stamp):
    """广播TF变换"""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])

    t.transform.rotation.x = float(rotation[0])
    t.transform.rotation.y = float(rotation[1])
    t.transform.rotation.z = float(rotation[2])
    t.transform.rotation.w = float(rotation[3])

    broadcaster.sendTransform(t)

class LidarVisualizer(Node):
    def __init__(self, mj_model, args):
        super().__init__('mujoco_lidar_test')
        self.site_name = "lidar_site"

        # 创建点云发布者
        self.pub_taichi = self.create_publisher(PointCloud2, '/lidar_points_taichi', 1)

        # 创建TF广播者
        self.tf_broadcaster = TransformBroadcaster(self)

        self.use_livox_lidar = False
        if args.lidar in {"avia", "mid40", "mid70", "mid360", "tele"}:
            self.livox_generator = scan_gen.LivoxGenerator(args.lidar)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.use_livox_lidar = True
        elif args.lidar == "HDL64":
            self.rays_theta, self.rays_phi = scan_gen.generate_HDL64()
        elif args.lidar == "vlp32":
            self.rays_theta, self.rays_phi = scan_gen.generate_vlp32()
        elif args.lidar == "os128":
            self.rays_theta, self.rays_phi = scan_gen.generate_os128()
        elif args.lidar == "custom":
            self.rays_theta, self.rays_phi = scan_gen.generate_grid_scan_pattern(360, 64, phi_range=(0., np.pi/2.))
        else:
            raise ValueError(f"不支持的LiDAR型号: {args.lidar}")

        # 优化内存布局
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        self.lidar = MjLidarWrapper(mj_model, site_name=self.site_name, backend="gpu")

        n_rays = len(self.rays_theta)
        _rays_phi = ti.ndarray(dtype=ti.f32, shape=n_rays)
        _rays_theta = ti.ndarray(dtype=ti.f32, shape=n_rays)
        _rays_phi.from_numpy(self.rays_phi)
        _rays_theta.from_numpy(self.rays_theta)
        self.rays_phi = _rays_phi
        self.rays_theta = _rays_theta

        self.get_logger().info(f"射线数量: {n_rays}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与ROS2集成')
    parser.add_argument('--lidar', type=str, default='mid360', help='LiDAR型号 (mid360, HDL64, vlp32, os128)', \
                        choices=['avia', 'HAP', 'horizon', 'mid40', 'mid70', 'mid360', 'tele', 'HDL64', 'vlp32', 'os128', 'custom'])
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    parser.add_argument('--rate', type=int, default=12, help='循环频率 (Hz) (默认: 12)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MuJoCo LiDAR可视化与ROS2集成")
    print("=" * 60)
    print(f"配置：")
    print(f"- LiDAR型号: {args.lidar}")
    print(f"- 循环频率: {args.rate} Hz")
    print(f"- 详细输出: {'启用' if args.verbose else '禁用'}")

    folder_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f"ros2 run rviz2 rviz2 -d {folder_path}/rviz_config/rviz2_config.rviz"
    print(f"在终端执行命令以开启rviz可视化:\n {cmd}")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_path("../models/mjcf/robocon2026.xml")
    mj_data = mujoco.MjData(mj_model)

    # 初始化ROS2
    rclpy.init()

    # 创建节点并运行
    node = LidarVisualizer(mj_model, args)

    spin_thread = threading.Thread(target=lambda:rclpy.spin(node))
    spin_thread.start()

    # 创建定时器
    step_cnt = 0
    render_fps = 60
    step_gap = render_fps // args.rate
    rate = node.create_rate(render_fps)

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            # 设置视图模式为site
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value

            lidar_pose = np.eye(4, dtype=np.float32)
            while rclpy.ok() and viewer.is_running:

                mj_model.body("lidar_base").pos[:] = site_position[:]
                mj_model.body("lidar_base").quat[:] = site_orientation[[3,0,1,2]]

                # 更新模拟
                for _ in range(int(1. / (render_fps * mj_model.opt.timestep))):
                    mujoco.mj_step(mj_model, mj_data)
                step_cnt += 1
                viewer.sync()
                rate.sleep()

                if step_cnt % step_gap == 0:

                    if node.use_livox_lidar:
                        node.rays_theta, node.rays_phi = node.livox_generator.sample_ray_angles()

                    # 获取激光雷达位姿
                    lidar_pose[:3, 3] = mj_data.site(node.site_name).xpos
                    lidar_pose[:3,:3] = mj_data.site(node.site_name).xmat.reshape(3,3)

                    start_time = time.time()
                    node.lidar.trace_rays(mj_data, node.rays_theta, node.rays_phi)
                    end_time = time.time()

                    points_local = node.lidar.get_hit_points()

                    # 获取激光雷达位置和方向
                    lidar_position = lidar_pose[:3,3]
                    lidar_orientation = Rotation.from_matrix(lidar_pose[:3,:3]).as_quat()

                    # 广播激光雷达的TF
                    broadcast_tf(node.tf_broadcaster, "world", "lidar", lidar_position, lidar_orientation, node.get_clock().now().to_msg())

                    # 发布点云
                    publish_point_cloud(node.pub_taichi, points_local, "lidar", node.get_clock().now().to_msg())

                    # 打印性能信息和当前位置
                    if args.verbose:
                        # 格式化欧拉角为度数
                        euler_deg = Rotation.from_quat(lidar_orientation).as_euler('xyz', degrees=True)
                        node.get_logger().info(f"位置: [{lidar_position[0]:.2f}, {lidar_position[1]:.2f}, {lidar_position[2]:.2f}], "
                            f"欧拉角: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°], "
                            f"耗时: {(end_time - start_time)*1000:.2f} ms")

    except KeyboardInterrupt:
        print("用户中断，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        spin_thread.join()
        node.destroy_node()
        rclpy.shutdown()
        print("程序结束")

if __name__ == "__main__":
    main()
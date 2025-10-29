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

        self.rays_theta, self.rays_phi = scan_gen.generate_airy96()

        # 优化内存布局
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
        # geomgroup[1] = 0  # 排除group 1中的几何体
        self.lidar = MjLidarWrapper(mj_model, site_name=self.site_name, backend="gpu", args={'bodyexclude': mj_model.body("mocap_body").id, "geomgroup":geomgroup})

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与ROS2集成')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出信息')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MuJoCo LiDAR可视化与ROS2集成")
    print("=" * 60)
    print(f"配置：")
    print(f"- LiDAR型号: Robosense airy-96")
    print(f"- 循环频率: 10 Hz")

    print("\n===================== LiDAR 模拟使用说明 =====================")
    print("1. 双击选中空中的绿色方块（模拟激光雷达的位置），")
    print("   按下ctrl，点击鼠标左键，拖动鼠标可以旋转绿色方块，")
    print("   按下ctrl和鼠标右键，拖动鼠标可以平移绿色方块")
    print("2. 按 Tab 键切换左侧 UI 的可视化界面；")
    print("   按 Shift+Tab 键切换右侧 UI 的可视化界面。")
    print("=" * 60)

    folder_path = os.path.dirname(os.path.abspath(__file__))
    cmd = f"rviz2 -d {folder_path}/rviz_config/rviz2_config.rviz"
    print(f"在终端执行命令以开启rviz可视化:\n {cmd}")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_path("../models/mjcf/mocap_env.xml")
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
    step_gap = render_fps // 12
    rate = node.create_rate(render_fps)

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            # 设置视图模式为site
            # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE.value
            # viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE.value

            lidar_pose = np.eye(4, dtype=np.float32)
            while rclpy.ok() and viewer.is_running():

                # 更新模拟
                for _ in range(int(1. / (render_fps * mj_model.opt.timestep))):
                    mujoco.mj_step(mj_model, mj_data)
                step_cnt += 1
                viewer.sync()
                rate.sleep()

                if step_cnt % step_gap == 0:

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
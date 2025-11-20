# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

import os
from etils import epath
import mujoco
import mujoco.viewer as viewer
import argparse
import numpy as np
import threading
# import taichi as ti
from scipy.spatial.transform import Rotation

from camera_utils import camera2k, get_site_tmat
from play_go2_joystick import OnnxController

import rclpy
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Twist
from sensor_msgs.msg import Image, CameraInfo, Imu, PointCloud2, PointField

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_MJCF_PATH = _HERE.parent.parent / "models" / "mjcf" / "scene_go2.xml"

_JOINT_NUM = 12
class OnnxControllerRos2(OnnxController, Node):
    """ONNX controller for the Go-2 robot."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
        lidar_type: str = "airy",
        #* add
        lidar_min_range: float = 1.0,
        lidar_max_range: float = 100.0,
        enable_camera: bool = True,
        tf_rate: float = 10.0,
        imu_rate: float = 200.0,
        image_rate: float = 20.0,
        lidar_rate: float = 20.0,
        caminfo_rate: float = 1.0,
    ):
        super().__init__(
            policy_path,
            default_angles,
            n_substeps,
            action_scale
        )
        Node.__init__(self, 'robocon_go2_node')

        self.camera_width = 640
        self.camera_height = 480
        self._camera_name = "head_camera"
        self._renderer = mujoco.Renderer(mj_model, height=self.camera_height, width=self.camera_width)
        
        self.enable_camera = enable_camera

        self.tf_rate = tf_rate
        self.imu_rate = imu_rate
        self.image_rate = image_rate
        self.lidar_rate = lidar_rate
        self.caminfo_rate = caminfo_rate

        self.init_topic_publisher(mj_model)

        # lidar
        self.lidar_min_range = lidar_min_range
        self.lidar_max_range = lidar_max_range
        self.dynamic_lidar = False
        if lidar_type == "airy":
            self.rays_theta, self.rays_phi = scan_gen.generate_airy96()
        elif lidar_type == "hdl64":
            self.rays_theta, self.rays_phi = scan_gen.generate_HDL64()
        elif lidar_type == "vlp32":
            self.rays_theta, self.rays_phi = scan_gen.generate_vlp32()
        elif lidar_type == "os128":
            self.rays_theta, self.rays_phi = scan_gen.generate_os128()
        elif lidar_type == "mid360":
            self.livox_generator = scan_gen.LivoxGenerator(lidar_type)
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            self.dynamic_lidar = True
        else:
            raise ValueError(f"Unknown lidar type: {lidar_type}")

        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)[::3]
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)[::3]

        geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
        geomgroup[3:] = 0  # 排除group 1中的几何体
        self.lidar = MjLidarWrapper(mj_model, site_name="lidar", backend="gpu", args={'bodyexclude': mj_model.body("base_0").id, "geomgroup":geomgroup})

        self.cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 10)
        self.latest_cmd_vel = np.zeros(2)

    def init_topic_publisher(self, mj_model):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_pub_time_tf = -1.

        self.imu_puber = self.create_publisher(Imu, '/imu', 10)
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "imu"
        self.last_pub_time_imu = -1.

        self.bridge = CvBridge()
        self.last_pub_time_image = -1.
        self.last_pub_time_caminfo = -1.

        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.pub_staticc_tf_once = False

        if self.enable_camera:
            self.head_color_puber = self.create_publisher(Image, '/head_camera/color/image_raw', 2)
            self.head_color_info_puber = self.create_publisher(CameraInfo, '/head_camera/color/camera_info', 2)
            self.head_color_info = CameraInfo()
            self.head_color_info.width = self.camera_width
            self.head_color_info.height = self.camera_height
            self.head_color_info.k = camera2k(mj_model.camera("head_camera").fovy.item() * np.pi / 180., self.camera_width, self.camera_height).flatten().tolist()

            self.head_depth_puber  = self.create_publisher(Image, '/head_camera/aligned_depth_to_color/image_raw', 2)
            self.head_depth_info_puber  = self.create_publisher(CameraInfo, '/head_camera/aligned_depth_to_color/camera_info', 2)
            self.head_depth_info = CameraInfo()
            self.head_depth_info.width = self.camera_width
            self.head_depth_info.height = self.camera_height
            self.head_depth_info.k = camera2k(mj_model.camera("head_camera").fovy.item() * np.pi / 180., self.camera_width, self.camera_height).flatten().tolist()

        self.lidar_puber = self.create_publisher(PointCloud2, '/lidar_points', 10)
        self.last_pub_time_lidar = -1.
        # 定义点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # 创建ROS2 PointCloud2消息
        pc_msg = PointCloud2()
        pc_msg.header.frame_id = "lidar"
        pc_msg.fields = fields
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12  # 3 个 float32 (x,y,z)
        pc_msg.height = 1
        pc_msg.is_dense = True
        self.pc_msg = pc_msg

    def get_obs(self, model, data) -> np.ndarray:
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:7+_JOINT_NUM] - self._default_angles
        joint_velocities = data.qvel[6:6+_JOINT_NUM]

        command = np.zeros(3, dtype=np.float32)
        command[0] = self.latest_cmd_vel[0]
        command[1] = 0.0
        command[2] = self.latest_cmd_vel[1]

        obs = np.hstack([
            linvel,
            gyro,
            gravity,
            joint_angles,
            joint_velocities,
            self._last_action,
            command
        ])
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().get_control(model, data)
        self.update_ros2(data)

    def publish_static_transform(self, mj_data, header_frame_id, child_frame_id):
        stfs_msg = TransformStamped()
        stfs_msg.header.stamp = self.get_clock().now().to_msg()
        stfs_msg.header.frame_id = header_frame_id
        stfs_msg.child_frame_id = child_frame_id

        tmat_base = get_site_tmat(mj_data, header_frame_id)
        tmat_child = get_site_tmat(mj_data, child_frame_id)
        tmat_trans = np.linalg.inv(tmat_base) @ tmat_child

        stfs_msg.transform.translation.x = tmat_trans[0, 3]
        stfs_msg.transform.translation.y = tmat_trans[1, 3]
        stfs_msg.transform.translation.z = tmat_trans[2, 3]

        quat = Rotation.from_matrix(tmat_trans[:3, :3]).as_quat()
        stfs_msg.transform.rotation.x = quat[0]
        stfs_msg.transform.rotation.y = quat[1]
        stfs_msg.transform.rotation.z = quat[2]
        stfs_msg.transform.rotation.w = quat[3]

        self.static_broadcaster.sendTransform(stfs_msg)

    def update_ros2(self, mj_data: mujoco.MjData) -> None:
        time_stamp = self.get_clock().now().to_msg()
        if not self.pub_staticc_tf_once:
            self.pub_staticc_tf_once = True
            self.publish_static_transform(mj_data, 'imu', 'lidar')
        self.publish_camera_info(mj_data)
        self.publish_tf(mj_data, time_stamp)
        self.publish_imu(mj_data, time_stamp)
        self.publish_images(mj_data, time_stamp)
        self.publish_lidar(mj_data, time_stamp)

    def publish_camera_info(self, mj_data):
        if not self.enable_camera:
            return
            
        if self.last_pub_time_caminfo > mj_data.time:
            self.last_pub_time_caminfo = mj_data.time
            return
        # 使用配置的帧率
        if mj_data.time - self.last_pub_time_caminfo < 1.0 / self.caminfo_rate:
            return
        self.last_pub_time_caminfo = mj_data.time
        self.head_color_info_puber.publish(self.head_color_info)
        self.head_depth_info_puber.publish(self.head_depth_info)

        self.publish_static_transform(mj_data, "imu", "lidar")

    def publish_tf(self, mj_data, time_stamp):
        if self.last_pub_time_tf > mj_data.time:
            self.last_pub_time_tf = mj_data.time
            return
        # 使用配置的帧率
        if mj_data.time - self.last_pub_time_tf < 1.0 / self.tf_rate:
            return
        self.last_pub_time_tf = mj_data.time

        trans_msg = TransformStamped()
        trans_msg.header.stamp = time_stamp
        trans_msg.header.frame_id = "odom"
        trans_msg.child_frame_id = "imu"
        trans_msg.transform.translation.x = mj_data.sensor("global_position").data[0]
        trans_msg.transform.translation.y = mj_data.sensor("global_position").data[1]
        trans_msg.transform.translation.z = mj_data.sensor("global_position").data[2]
        trans_msg.transform.rotation.w = mj_data.sensor("orientation").data[0]
        trans_msg.transform.rotation.x = mj_data.sensor("orientation").data[1]
        trans_msg.transform.rotation.y = mj_data.sensor("orientation").data[2]
        trans_msg.transform.rotation.z = mj_data.sensor("orientation").data[3]
        self.tf_broadcaster.sendTransform(trans_msg)

    def publish_imu(self, mj_data, time_stamp):
        if self.last_pub_time_imu > mj_data.time:
            self.last_pub_time_imu = mj_data.time
            return
        # 使用配置的帧率
        if mj_data.time - self.last_pub_time_imu < 1.0 / self.imu_rate: # TODO fps Bug
            return
        self.last_pub_time_imu = mj_data.time

        self.imu_msg.header.stamp = time_stamp
        self.imu_msg.orientation.w = mj_data.sensor("orientation").data[0]
        self.imu_msg.orientation.x = mj_data.sensor("orientation").data[1]
        self.imu_msg.orientation.y = mj_data.sensor("orientation").data[2]
        self.imu_msg.orientation.z = mj_data.sensor("orientation").data[3]
        self.imu_msg.angular_velocity.x = mj_data.sensor("gyro").data[0]
        self.imu_msg.angular_velocity.y = mj_data.sensor("gyro").data[1]
        self.imu_msg.angular_velocity.z = mj_data.sensor("gyro").data[2]
        self.imu_msg.linear_acceleration.x = mj_data.sensor("accelerometer").data[0]
        self.imu_msg.linear_acceleration.y = mj_data.sensor("accelerometer").data[1]
        self.imu_msg.linear_acceleration.z = mj_data.sensor("accelerometer").data[2]
        self.imu_puber.publish(self.imu_msg)

    def publish_images(self, mj_data, time_stamp):
        if not self.enable_camera:
            return
            
        if self.last_pub_time_image > mj_data.time:
            self.last_pub_time_image = mj_data.time
            return
        if mj_data.time - self.last_pub_time_image < 1.0 / self.image_rate: # TODO fps Bug
            return
        self.last_pub_time_image = mj_data.time

        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(mj_data, self._camera_name)
        head_color_img_msg = self.bridge.cv2_to_imgmsg(self._renderer.render(), encoding="rgb8")
        head_color_img_msg.header.stamp = time_stamp
        head_color_img_msg.header.frame_id = "head_camera"
        self.head_color_puber.publish(head_color_img_msg)

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(mj_data, self._camera_name)
        head_depth_img = np.array(np.clip(self._renderer.render()*1e3, 0, 65535), dtype=np.uint16)
        head_depth_img_msg = self.bridge.cv2_to_imgmsg(head_depth_img, encoding="mono16")
        head_depth_img_msg.header.stamp = time_stamp
        head_depth_img_msg.header.frame_id = "head_camera"
        self.head_depth_puber.publish(head_depth_img_msg)

    def publish_lidar(self, mj_data, time_stamp):
        if self.last_pub_time_lidar > mj_data.time:
            self.last_pub_time_lidar = mj_data.time
            return
        if mj_data.time - self.last_pub_time_lidar < 1. / self.lidar_rate:
            return
        self.last_pub_time_lidar = mj_data.time

        if self.dynamic_lidar:
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        self.lidar.trace_rays(mj_data, self.rays_theta, self.rays_phi)
        points = self.lidar.get_hit_points()

        if points.size > 0:
            distances = np.sqrt(np.sum(points**2, axis=1))
            
            range_mask = (distances >= self.lidar_min_range) & (distances <= self.lidar_max_range)
            filtered_points = points[range_mask]
        else:
            filtered_points = points

        self.pc_msg.header.stamp = time_stamp
        self.pc_msg.row_step = self.pc_msg.point_step * filtered_points.shape[0]
        self.pc_msg.width = filtered_points.shape[0]
        self.pc_msg.data = filtered_points.tobytes()

        self.lidar_puber.publish(self.pc_msg)

    def cmd_vel_callback(self, msg):
        self.latest_cmd_vel[0] = msg.linear.x
        self.latest_cmd_vel[1] = msg.angular.z

def load_callback(model=None, data=None):
    global args
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        _MJCF_PATH.as_posix()
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.001
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy = OnnxControllerRos2(
        model,
        policy_path=(_ONNX_DIR / "go2_policy.onnx").as_posix(),
        default_angles=np.array(model.keyframe("home").qpos[7:7+_JOINT_NUM]),
        n_substeps=n_substeps,
        action_scale=0.5,
        lidar_type=args.lidar,
        enable_camera=False,
    )

    spin_thread = threading.Thread(target=lambda:rclpy.spin(policy), daemon=True)
    spin_thread.start()

    mujoco.set_mjcb_control(policy.get_control)

    return model, data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MuJoCo LiDAR可视化与ROS2集成')
    parser.add_argument(
        "--lidar",
        type=str,
        default="airy",
        help="LiDAR型号 (airy, mid360)",
        choices=["airy", "mid360", "hdl64", "vlp32", "os128"],
    )
    args = parser.parse_args()

    rclpy.init()

    print("=" * 60)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    rviz_config = os.path.join(folder_path, "../rviz_config/go2.rviz")
    # 尝试自动启动 rviz2（非阻塞）。如果系统没有 rviz2 则打印提示。
    import subprocess
    import shutil

    rviz_proc = None
    rviz_exec = shutil.which("rviz2")
    if rviz_exec is None:
        print("rviz2 未找到，请手动安装或手动启动 rviz2 来查看话题。")
    else:
        try:
            rviz_proc = subprocess.Popen([rviz_exec, "-d", rviz_config])
            print(f"启动 rviz2 (pid={rviz_proc.pid})，使用配置: {rviz_config}")
        except Exception as e:
            print(f"自动启动 rviz2 失败: {e}")

    try:
        viewer.launch(loader=load_callback)
    finally:
        # 仿真退出时尝试优雅关闭 rviz2
        if rviz_proc is not None:
            try:
                if rviz_proc.poll() is None:
                    rviz_proc.terminate()
                    rviz_proc.wait(timeout=2)
            except Exception:
                try:
                    rviz_proc.kill()
                except Exception:
                    pass

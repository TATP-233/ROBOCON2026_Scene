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
import numpy as np
import threading
# import taichi as ti
from scipy.spatial.transform import Rotation

from camera_utils import camera2k, get_site_tmat
from play_g1_joystick import OnnxController

import rclpy
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, Imu, PointCloud2, PointField

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_MJCF_PATH = _HERE.parent.parent / "models" / "mjcf" / "scene_g1.xml"

_JOINT_NUM = 29
class OnnxControllerRos2(OnnxController, Node):
    """ONNX controller for the G-1 robot."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        policy_path: str,
        default_angles: np.ndarray,
        ctrl_dt: float,
        n_substeps: int,
        action_scale: float = 0.5,
    ):
        super().__init__(
            policy_path,
            default_angles,
            ctrl_dt,
            n_substeps,
            action_scale
        )
        Node.__init__(self, 'robocon_g1_node')

        self.camera_width = 640
        self.camera_height = 480
        self._camera_name = "head_camera"
        self._renderer = mujoco.Renderer(mj_model, height=self.camera_height, width=self.camera_width)

        self.init_topic_publisher(mj_model)

        # lidar
        self.rays_theta, self.rays_phi = scan_gen.generate_airy96()
        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)[::3]
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)[::3]

        geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
        geomgroup[3:] = 0  # 排除group 1中的几何体
        self.lidar = MjLidarWrapper(mj_model, site_name="lidar", backend="gpu", args={'bodyexclude': -1, "geomgroup":geomgroup})

    def init_topic_publisher(self, mj_model):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.last_pub_time_tf = -1.

        self.imu_puber = self.create_publisher(Imu, '/imu', 5)
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "imu"
        self.last_pub_time_imu = -1.

        self.bridge = CvBridge()
        self.last_pub_time_image = -1.
        self.last_pub_time_caminfo = -1.

        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.pub_staticc_tf_once = False

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

        self.lidar_puber = self.create_publisher(PointCloud2, '/lidar_points', 1)
        self.last_pub_time_lidar = -1.
        # 定义点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # 创建ROS2 PointCloud2消息
        pc_msg = PointCloud2()
        pc_msg.header.frame_id = 'lidar'
        pc_msg.fields = fields
        pc_msg.is_bigendian = False
        pc_msg.point_step = 12  # 3 个 float32 (x,y,z)
        pc_msg.height = 1
        pc_msg.is_dense = True
        self.pc_msg = pc_msg

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
            self.publish_static_transform(mj_data, 'imu_in_pelvis', 'lidar')
        self.publish_camera_info(mj_data)
        self.publish_tf(mj_data, time_stamp)
        self.publish_imu(mj_data, time_stamp)
        self.publish_images(mj_data, time_stamp)
        self.publish_lidar(mj_data, time_stamp)

    def publish_camera_info(self, mj_data):
        if self.last_pub_time_caminfo > mj_data.time:
            self.last_pub_time_caminfo = mj_data.time
            return
        if mj_data.time - self.last_pub_time_caminfo < 1.0:
            return
        self.last_pub_time_caminfo = mj_data.time
        self.head_color_info_puber.publish(self.head_color_info)
        self.head_depth_info_puber.publish(self.head_depth_info)
    
    def publish_tf(self, mj_data, time_stamp):
        if self.last_pub_time_tf > mj_data.time:
            self.last_pub_time_tf = mj_data.time
            return
        if mj_data.time - self.last_pub_time_tf < 1. / 10.:
            return
        self.last_pub_time_tf = mj_data.time

        trans_msg = TransformStamped()
        trans_msg.header.stamp = time_stamp
        trans_msg.header.frame_id = "odom"
        trans_msg.child_frame_id = "imu_in_pelvis"
        trans_msg.transform.translation.x = mj_data.sensor("position").data[0]
        trans_msg.transform.translation.y = mj_data.sensor("position").data[1]
        trans_msg.transform.translation.z = mj_data.sensor("position").data[2]
        trans_msg.transform.rotation.w = mj_data.sensor("orientation_pelvis").data[0]
        trans_msg.transform.rotation.x = mj_data.sensor("orientation_pelvis").data[1]
        trans_msg.transform.rotation.y = mj_data.sensor("orientation_pelvis").data[2]
        trans_msg.transform.rotation.z = mj_data.sensor("orientation_pelvis").data[3]
        self.tf_broadcaster.sendTransform(trans_msg)

    def publish_imu(self, mj_data, time_stamp):
        if self.last_pub_time_imu > mj_data.time:
            self.last_pub_time_imu = mj_data.time
            return
        if mj_data.time - self.last_pub_time_imu < 0.5 / 250.: # TODO fps Bug
            return
        self.last_pub_time_imu = mj_data.time

        self.imu_msg.header.stamp = time_stamp
        self.imu_msg.orientation.w = mj_data.sensor("orientation_pelvis").data[0]
        self.imu_msg.orientation.x = mj_data.sensor("orientation_pelvis").data[1]
        self.imu_msg.orientation.y = mj_data.sensor("orientation_pelvis").data[2]
        self.imu_msg.orientation.z = mj_data.sensor("orientation_pelvis").data[3]
        self.imu_msg.angular_velocity.x = mj_data.sensor("gyro_pelvis").data[0]
        self.imu_msg.angular_velocity.y = mj_data.sensor("gyro_pelvis").data[1]
        self.imu_msg.angular_velocity.z = mj_data.sensor("gyro_pelvis").data[2]
        self.imu_msg.linear_acceleration.x = mj_data.sensor("accelerometer_pelvis").data[0]
        self.imu_msg.linear_acceleration.y = mj_data.sensor("accelerometer_pelvis").data[1]
        self.imu_msg.linear_acceleration.z = mj_data.sensor("accelerometer_pelvis").data[2]
        self.imu_puber.publish(self.imu_msg)

    def publish_images(self, mj_data, time_stamp):
        if self.last_pub_time_image > mj_data.time:
            self.last_pub_time_image = mj_data.time
            return
        if mj_data.time - self.last_pub_time_image < 0.5 / 20.: # TODO fps Bug
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
        if mj_data.time - self.last_pub_time_lidar < 1. / 10.:
            return
        self.last_pub_time_lidar = mj_data.time

        self.lidar.trace_rays(mj_data, self.rays_theta, self.rays_phi)
        points = self.lidar.get_hit_points()

        self.pc_msg.header.stamp = time_stamp
        self.pc_msg.row_step = self.pc_msg.point_step * points.shape[0]
        self.pc_msg.width = points.shape[0]
        self.pc_msg.data = points.tobytes()

        self.lidar_puber.publish(self.pc_msg)

def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        _MJCF_PATH.as_posix()
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.002
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy = OnnxControllerRos2(
        model,
        policy_path=(_ONNX_DIR / "g1_policy.onnx").as_posix(),
        default_angles=np.array(model.keyframe("knees_bent").qpos[7:7+_JOINT_NUM]),
        ctrl_dt=ctrl_dt,
        n_substeps=n_substeps,
        action_scale=0.5,
    )

    spin_thread = threading.Thread(target=lambda:rclpy.spin(policy), daemon=True)
    spin_thread.start()

    mujoco.set_mjcb_control(policy.get_control)

    return model, data

if __name__ == "__main__":
    rclpy.init()

    print("=" * 60)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    rviz_config = os.path.join(folder_path, "../rviz_config/g1.rviz")
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

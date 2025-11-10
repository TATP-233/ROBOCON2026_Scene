# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import collections
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

import queue as pyqueue
import multiprocessing as mp
from gamepad_reader import joystick_process_main

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_MJCF_PATH = _HERE.parent.parent / "models" / "mjcf" / "scene_g1.xml"

_JOINT_NUM = 29
class OnnxController:
    """ONNX controller for the Go-1 robot."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
    ):
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._policy.get_inputs()[0].name
        self._output_name = self._policy.get_outputs()[0].name

        self._default_angles = default_angles

        self._num_actions = 15 # 12
        # self._idx_dof = np.concatenate([list(range(13)), list(range(15, 29))])
        self._last_action = np.zeros((self._num_actions, ), dtype=np.float32)
        self._num_single_obs = 83  # command(4) + gyro(3) + gravity(3) + joint_pos(29) + joint_vel(29) + last_action(15)
        self._obs_history_len = 6
        self._num_obs = self._num_single_obs * self._obs_history_len  # 83 * 6 (observation dimension * history length)

        self.obs = np.zeros(self._num_obs, dtype=np.float32)
        self.obs_history = collections.deque(maxlen=self._obs_history_len)
        for _ in range(self._obs_history_len):
            self.obs_history.append(np.zeros(self._num_single_obs, dtype=np.float32))

        self._counter = 0
        self._n_substeps = n_substeps

        self._scale_command = np.array([2.0, 2.0, 0.25, 1.0], dtype=np.float32)
        self._scale_gyro = 0.25
        self._scale_dof_pos = 1.0
        self._scale_dof_vel = 0.05
        self._scale_action = 0.25


        self.target_height = 0.7

        self.joy_queue = mp.Queue(maxsize=1)
        joy_stop_event = mp.Event()
        self.joy_process = mp.Process(target=joystick_process_main, args=(self.joy_queue, joy_stop_event), daemon=True)
        self.joy_process.start()
        self.latest_axes, self.latest_buttons = None, None

    def get_obs(self, model, data) -> np.ndarray:
        gyro = data.sensor("gyro_pelvis").data
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:7+_JOINT_NUM] - self._default_angles
        joint_velocities = data.qvel[6:6+_JOINT_NUM]

        command = np.zeros(4, dtype=np.float32)
        if not self.joy_queue is None:
            try:
                self.latest_axes, self.latest_buttons = self.joy_queue.get_nowait()
                command[:3] = -np.array([self.latest_axes[1] * 1., self.latest_axes[0] * 0.5, self.latest_axes[3] * 0.8])
                self.target_height += (self.latest_axes[2] - self.latest_axes[5]) * self._n_substeps * model.opt.timestep * 0.1
                self.target_height = np.clip(self.target_height, 0.24, 0.74)
                command[3] = self.target_height
            except pyqueue.Empty:
                pass

        obs = np.hstack([
            command * self._scale_command,  # 0:4   4
            gyro * self._scale_gyro,        # 4:7   3
            gravity,                        # 7:10  3
            joint_angles * self._scale_dof_pos,         # 10:37 27
            joint_velocities * self._scale_dof_vel,     # 37:64 27
            # _joint_angles * self._scale_dof_pos,         # 10:37 27
            # _joint_velocities * self._scale_dof_vel,     # 37:64 27
            self._last_action,     # 64:76 12
        ])
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            single_obs = self.get_obs(model, data)
            self.obs_history.append(single_obs)

            for i, hist_obs in enumerate(self.obs_history):
                start_idx = i * single_obs.shape[0]
                end_idx = start_idx + single_obs.shape[0]
                self.obs[start_idx:end_idx] = hist_obs
            onnx_input = {self._input_name: self.obs.reshape(1, -1)}
            onnx_pred = self._policy.run([self._output_name], onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:self._num_actions] = onnx_pred * self._scale_action + self._default_angles[:self._num_actions]

def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        _MJCF_PATH.as_posix(),
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.002
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy = OnnxController(
        policy_path=(_ONNX_DIR / "g1_wb_policy_v2.onnx").as_posix(),
        default_angles=np.array(model.keyframe("homie").qpos[7:7+_JOINT_NUM]),
        n_substeps=n_substeps,
    )

    mujoco.set_mjcb_control(policy.get_control)

    return model, data

if __name__ == "__main__":
    viewer.launch(loader=load_callback)

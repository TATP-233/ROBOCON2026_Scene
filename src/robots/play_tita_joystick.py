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
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

import queue as pyqueue
import multiprocessing as mp
from gamepad_reader import joystick_process_main

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_MJCF_PATH = _HERE.parent.parent / "models" / "mjcf" / "scene_tita.xml"

_JOINT_NUM = 8
class OnnxController:
    """ONNX controller for the Go-1 robot."""
    _joint_kp = np.array([36.0, 36.0, 36.0, 12.0, 36.0, 36.0, 36.0, 12.0])
    _joint_kd = np.array([1.2, 1.2, 1.2, 0.6, 1.2, 1.2, 1.2, 0.5])

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
    ):
        self._output_names = ["88"]
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)
        self._target_q = default_angles.copy()

        self._counter = 0
        self._n_substeps = n_substeps
        
        self._history_len = 10
        self._num_obs = 33
        self._obs_history = np.zeros((self._history_len, self._num_obs), dtype=np.float32)
        
        self._filtered_gyro = np.zeros(3, dtype=np.float32)

        self.joy_queue = mp.Queue(maxsize=1)
        joy_stop_event = mp.Event()
        self.joy_process = mp.Process(target=joystick_process_main, args=(self.joy_queue, joy_stop_event), daemon=True)
        self.joy_process.start()
        self.latest_axes, self.latest_buttons = None, None

    def get_obs(self, model, data) -> np.ndarray:
        # linvel = data.sensor("local_linvel_pelvis").data
        gyro = data.sensor("trunk_gyro").data
        self._filtered_gyro = 0.97 * gyro + 0.03 * self._filtered_gyro
        
        imu_xmat = data.site_xmat[model.site("trunk_imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:7+_JOINT_NUM] - self._default_angles
        joint_velocities = data.qvel[6:6+_JOINT_NUM]

        command = np.zeros(3, dtype=np.float32)
        if not self.joy_queue is None:
            try:
                self.latest_axes, self.latest_buttons = self.joy_queue.get_nowait()
                command = -np.array([self.latest_axes[1] * 1., self.latest_axes[0] * 0., self.latest_axes[3] * 1.])
            except pyqueue.Empty:
                pass

        obs = np.hstack([
            self._filtered_gyro * 0.25,
            gravity,
            command * np.array([2.0, 2.0, 0.25]),
            joint_angles * 1.0,
            joint_velocities * 0.05,
            self._last_action,
        ])
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            
            # Update history
            self._obs_history = np.roll(self._obs_history, -1, axis=0)
            self._obs_history[-1] = obs

            onnx_input = {
                "0": obs.reshape(1, -1),
                "obs_hist": self._obs_history.reshape(1, self._history_len, self._num_obs)
            }
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            
            # Apply hip scale reduction
            action_scaled = onnx_pred * self._action_scale
            hip_indices = [0, 4]
            action_scaled[hip_indices] *= 0.5
            
            self._target_q = action_scaled + self._default_angles
            
        data.ctrl[:] = self._joint_kp * (self._target_q - data.qpos[7:7+_JOINT_NUM]) - self._joint_kd * data.qvel[6:6+_JOINT_NUM]

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
        policy_path=(_ONNX_DIR / "tita_policy.onnx").as_posix(),
        default_angles=np.array(model.keyframe("home").qpos[7:7+_JOINT_NUM]),
        n_substeps=n_substeps,
        action_scale=0.25,
    )

    mujoco.set_mjcb_control(policy.get_control)

    return model, data

if __name__ == "__main__":
    viewer.launch(loader=load_callback)

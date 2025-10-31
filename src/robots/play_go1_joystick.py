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
_MJCF_PATH = _HERE.parent.parent / "models" / "mjcf" / "scene_go1.xml"

_JOINT_NUM = 12
class OnnxController:
    """ONNX controller for the Go-1 robot."""

    def __init__(
        self,
        policy_path: str,
        default_angles: np.ndarray,
        n_substeps: int,
        action_scale: float = 0.5,
    ):
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(
            policy_path, providers=["CPUExecutionProvider"]
        )

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        self.joy_queue = mp.Queue(maxsize=1)
        joy_stop_event = mp.Event()
        self.joy_process = mp.Process(target=joystick_process_main, args=(self.joy_queue, joy_stop_event), daemon=True)
        self.joy_process.start()
        self.latest_axes, self.latest_buttons = None, None

    def get_obs(self, model, data) -> np.ndarray:
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:7+_JOINT_NUM] - self._default_angles
        joint_velocities = data.qvel[6:6+_JOINT_NUM]

        command = np.zeros(3, dtype=np.float32)
        if not self.joy_queue is None:
            try:
                self.latest_axes, self.latest_buttons = self.joy_queue.get_nowait()
                command = -np.array([self.latest_axes[1] * 1.5, self.latest_axes[0] * 0.8, self.latest_axes[3] * 2 * np.pi])
            except pyqueue.Empty:
                pass

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
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_input = {"obs": obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles

def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        _MJCF_PATH.as_posix()
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    model.opt.timestep = sim_dt

    policy = OnnxController(
        policy_path=(_ONNX_DIR / "go1_policy.onnx").as_posix(),
        default_angles=np.array(model.keyframe("home_red").qpos[7:7+_JOINT_NUM]),
        n_substeps=n_substeps,
        action_scale=0.5,
    )

    mujoco.set_mjcb_control(policy.get_control)

    return model, data

if __name__ == "__main__":
    viewer.launch(loader=load_callback)

# Copyright 2025
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
"""Deploy an ONNX policy for SF_TRON1A robot in MuJoCo."""

import os
import sys
import yaml
import numpy as np
from etils import epath

from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer as viewer
import onnxruntime as ort
import queue as pyqueue
import multiprocessing as mp
from gamepad_reader import joystick_process_main

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx" / "sf_tron_a1"
_MJCF_PATH = _HERE.parent.parent / "models" / "mjcf" / "scene_tron.xml"

class OnnxSolefootController:
    """ONNX controller for the Solefoot robot."""

    def __init__(self):

        # Load configuration and model file paths
        self.config_file = f'{_ONNX_DIR.as_posix()}/params.yaml'
        self.model_policy = f'{_ONNX_DIR.as_posix()}/policy.onnx'
        self.model_encoder = f'{_ONNX_DIR.as_posix()}/encoder.onnx'

        # Load configuration settings from the YAML file
        self.load_config(self.config_file)
        
        # Load the ONNX models
        self.initialize_onnx_models()

        # Initialize robot state variables
        self.robot_state_q = np.zeros(self.joint_num)
        self.robot_state_dq = np.zeros(self.joint_num)
        self.robot_state_tau = np.zeros(self.joint_num)

        # Initialize IMU data
        self.imu_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.imu_gyro = np.zeros(3)
        self.imu_acc = np.zeros(3)

        # Initialize control variables
        self.robot_cmd_q = np.zeros(self.joint_num)
        self.robot_cmd_dq = np.zeros(self.joint_num)
        self.robot_cmd_tau = np.zeros(self.joint_num)
        self.robot_cmd_Kp = np.array([self.control_cfg['stiffness']] * self.joint_num)
        self.robot_cmd_Kd = np.array([self.control_cfg['damping']] * self.joint_num)

        # Initialize mode and counters
        self.mode = "WALK"
        self.loop_count = 0
        self.stand_percent = 0.0
        self.gait_index = 0.0
        self.is_first_rec_obs = True

        # Initialize joystick
        self.joy_queue = mp.Queue(maxsize=1)
        joy_stop_event = mp.Event()
        self.joy_process = mp.Process(
            target=joystick_process_main, 
            args=(self.joy_queue, joy_stop_event), 
            daemon=True
        )
        self.joy_process.start()
        self.latest_axes, self.latest_buttons = None, None

        # Counter for decimation
        self._counter = 0
        self._n_substeps = self.control_cfg['decimation']

    def initialize_onnx_models(self):
        """Initialize ONNX Runtime sessions for policy and encoder."""
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False

        cpu_providers = ['CPUExecutionProvider']
        
        # Load policy model
        self.policy_session = ort.InferenceSession(
            self.model_policy, 
            sess_options=session_options, 
            providers=cpu_providers
        )
        self.policy_input_names = [inp.name for inp in self.policy_session.get_inputs()]
        self.policy_output_names = [out.name for out in self.policy_session.get_outputs()]

        # Load encoder model
        self.encoder_session = ort.InferenceSession(
            self.model_encoder, 
            sess_options=session_options, 
            providers=cpu_providers
        )
        self.encoder_input_names = [inp.name for inp in self.encoder_session.get_inputs()]
        self.encoder_output_names = [out.name for out in self.encoder_session.get_outputs()]

    def load_config(self, config_file):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        cfg = config['PointfootCfg']
        self.joint_names = cfg['joint_names']
        self.init_state = cfg['init_state']['default_joint_angle']
        self.stand_duration = cfg['stand_mode']['stand_duration']
        self.control_cfg = cfg['control']
        self.rl_cfg = cfg['normalization']
        self.obs_scales = cfg['normalization']['obs_scales']
        self.actions_size = cfg['size']['actions_size']
        self.commands_size = cfg['size']['commands_size']
        self.observations_size = cfg['size']['observations_size']
        self.obs_history_length = cfg['size']['obs_history_length']
        self.encoder_output_size = cfg['size']['encoder_output_size']
        self.imu_orientation_offset = np.array(list(cfg['imu_orientation_offset'].values()))
        self.user_cmd_cfg = cfg['user_cmd_scales']
        self.loop_frequency = cfg['loop_frequency']
        self.encoder_input_size = self.obs_history_length * self.observations_size

        # Initialize variables
        self.proprio_history_vector = np.zeros(self.obs_history_length * self.observations_size)
        self.encoder_out = np.zeros(self.encoder_output_size)
        self.actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(self.commands_size)
        self.scaled_commands = np.zeros(self.commands_size)
        self.joint_num = len(self.joint_names)

        self.ankle_joint_damping = self.control_cfg['ankle_joint_damping']
        self.ankle_joint_torque_limit = self.control_cfg['ankle_joint_torque_limit']
        self.gait_frequencies = cfg['gait']['frequencies']
        self.gait_swing_height = cfg['gait']['swing_height']

        # Initialize joint angles
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]

        # Default joint angles for standing
        self.default_joint_angles = np.array([0.0] * len(self.joint_names))

    def get_joystick_command(self):
        """Get command from joystick."""
        command = np.zeros(3, dtype=np.float32)
        if self.joy_queue is not None:
            try:
                self.latest_axes, self.latest_buttons = self.joy_queue.get_nowait()
                # Map joystick axes to robot commands
                linear_x = -self.latest_axes[1]  # Forward/backward
                linear_y = -self.latest_axes[0]  # Left/right
                angular_z = -self.latest_axes[3]  # Rotation
                
                # Clamp values
                linear_x = max(-1.0, min(1.0, linear_x))
                linear_y = max(-1.0, min(1.0, linear_y))
                angular_z = max(-1.0, min(1.0, angular_z))
                
                # command = np.array([linear_x * 0.5, linear_y * 0.5, angular_z * 0.5])
                command = np.array([linear_x, linear_y, angular_z])
            except pyqueue.Empty:
                pass
        return command

    def compute_observation(self):
        """Compute observation from robot state and IMU data."""
        # Convert IMU orientation from quaternion to Euler angles
        imu_orientation = self.imu_quat
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()

        # Project gravity vector into body frame
        gravity_vector = np.array([0, 0, -1])
        projected_gravity = np.dot(inverse_rot, gravity_vector)

        # Get base angular velocity
        base_ang_vel = self.imu_gyro.copy()
        
        # Apply IMU orientation offset correction
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()
        base_ang_vel = np.dot(rot, base_ang_vel)
        projected_gravity = np.dot(rot, projected_gravity)

        # Get joint positions and velocities
        joint_positions = self.robot_state_q.copy()
        joint_velocities = self.robot_state_dq.copy()

        # Gait parameters
        gait = np.array([self.gait_frequencies, 0.5, 0.5, self.gait_swing_height])
        self.gait_index += 0.02 * gait[0]
        if self.gait_index > 1.0:
            self.gait_index = 0.0
        gait_clock = np.array([np.sin(self.gait_index * 2 * np.pi), np.cos(self.gait_index * 2 * np.pi)])

        # Get last actions
        actions = self.last_actions.copy()

        # Scale commands
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],
            self.user_cmd_cfg['lin_vel_y'],
            self.user_cmd_cfg['ang_vel_yaw'],
            1.0, 1.0
        ])
        self.scaled_commands = np.dot(command_scaler, self.commands)

        # Compute joint position input
        joint_pos_input = (joint_positions - self.init_joint_angles) * self.obs_scales['dof_pos']
        
        # Create observation vector
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],
            projected_gravity,
            joint_pos_input,
            joint_velocities * self.obs_scales['dof_vel'],
            actions,
            gait_clock,
            gait
        ])

        # Initialize proprioceptive history on first observation
        if self.is_first_rec_obs:
            self.proprio_history_buffer = np.zeros(self.encoder_input_size)
            for i in range(self.obs_history_length):
                self.proprio_history_buffer[i * self.observations_size:(i + 1) * self.observations_size] = obs
            self.is_first_rec_obs = False
        
        # Update proprioceptive history buffer
        self.proprio_history_buffer[:-self.observations_size] = self.proprio_history_buffer[self.observations_size:]
        self.proprio_history_buffer[-self.observations_size:] = obs
        self.proprio_history_vector = np.array(self.proprio_history_buffer)

        # Clip observations
        self.observations = np.clip(
            obs, 
            -self.rl_cfg['clip_scales']['clip_observations'],
            self.rl_cfg['clip_scales']['clip_observations']
        )

    def compute_encoder(self):
        """Compute encoder output."""
        input_tensor = self.proprio_history_buffer.astype(np.float32)
        inputs = {self.encoder_input_names[0]: input_tensor}
        output = self.encoder_session.run(self.encoder_output_names, inputs)
        self.encoder_out = np.array(output).flatten()

    def compute_actions(self):
        """Compute actions using policy network."""
        input_tensor = np.concatenate([
            self.encoder_out, 
            self.observations, 
            self.scaled_commands
        ], axis=0).astype(np.float32)
        
        inputs = {self.policy_input_names[0]: input_tensor}
        output = self.policy_session.run(self.policy_output_names, inputs)
        self.actions = np.array(output).flatten()

    def handle_stand_mode(self):
        """Handle stand mode transition."""
        if self.stand_percent < 1:
            for j in range(len(self.joint_names)):
                # Interpolate between default and initial joint angles
                pos_des = (self.default_joint_angles[j] * (1 - self.stand_percent) + 
                          self.init_state[self.joint_names[j]] * self.stand_percent)
                self.robot_cmd_q[j] = pos_des
                self.robot_cmd_dq[j] = 0
                self.robot_cmd_tau[j] = 0
                self.robot_cmd_Kp[j] = self.control_cfg['stiffness']
                self.robot_cmd_Kd[j] = self.control_cfg['damping']
            
            self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        else:
            self.mode = "WALK"

    def handle_walk_mode(self):
        """Handle walk mode with RL policy."""
        # Execute actions every decimation steps
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_encoder()
            self.compute_actions()
            
            # Clip actions
            action_min = -self.rl_cfg['clip_scales']['clip_actions']
            action_max = self.rl_cfg['clip_scales']['clip_actions']
            self.actions = np.clip(self.actions, action_min, action_max)

        # Set joint commands
        joint_pos = self.robot_state_q.copy()
        joint_vel = self.robot_state_dq.copy()

        for i in range(len(joint_pos)):
            if (i + 1) % 4 != 0:  # Non-ankle joints
                # Compute action limits based on torque limits
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                            (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                            self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                            (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                            self.control_cfg['stiffness'])

                # Clip action
                clipped_action = max(action_min / self.control_cfg['action_scale_pos'],
                                   min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))

                # Compute desired position
                pos_des = clipped_action * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                
                self.robot_cmd_q[i] = pos_des
                self.robot_cmd_dq[i] = 0
                self.robot_cmd_tau[i] = 0
                self.robot_cmd_Kp[i] = self.control_cfg['stiffness']
                self.robot_cmd_Kd[i] = self.control_cfg['damping']
                self.last_actions[i] = self.actions[i]
            else:  # Ankle joints
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                            (self.ankle_joint_damping * joint_vel[i] - self.ankle_joint_torque_limit) /
                            self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                            (self.ankle_joint_damping * joint_vel[i] + self.ankle_joint_torque_limit) /
                            self.control_cfg['stiffness'])
                
                clipped_action = max(action_min / self.control_cfg['action_scale_pos'],
                                   min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))
                
                pos_des = clipped_action * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                
                self.robot_cmd_q[i] = pos_des
                self.robot_cmd_dq[i] = 0
                self.robot_cmd_tau[i] = 0
                self.robot_cmd_Kp[i] = self.control_cfg['stiffness']
                self.robot_cmd_Kd[i] = self.ankle_joint_damping
                self.last_actions[i] = self.actions[i]

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Main control function called by MuJoCo."""
        # Update robot state from MuJoCo
        for i in range(self.joint_num):
            self.robot_state_q[i] = data.qpos[7 + i]
            self.robot_state_dq[i] = data.qvel[6 + i]
            self.robot_state_tau[i] = data.ctrl[i]

        # Update IMU data from MuJoCo
        imu_quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
        self.imu_quat[0] = data.sensordata[model.sensor_adr[imu_quat_id] + 1]
        self.imu_quat[1] = data.sensordata[model.sensor_adr[imu_quat_id] + 2]
        self.imu_quat[2] = data.sensordata[model.sensor_adr[imu_quat_id] + 3]
        self.imu_quat[3] = data.sensordata[model.sensor_adr[imu_quat_id] + 0]

        imu_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        self.imu_gyro[0] = data.sensordata[model.sensor_adr[imu_gyro_id] + 0]
        self.imu_gyro[1] = data.sensordata[model.sensor_adr[imu_gyro_id] + 1]
        self.imu_gyro[2] = data.sensordata[model.sensor_adr[imu_gyro_id] + 2]

        imu_acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")
        self.imu_acc[0] = data.sensordata[model.sensor_adr[imu_acc_id] + 0]
        self.imu_acc[1] = data.sensordata[model.sensor_adr[imu_acc_id] + 1]
        self.imu_acc[2] = data.sensordata[model.sensor_adr[imu_acc_id] + 2]

        # Get joystick commands
        joy_cmd = self.get_joystick_command()
        self.commands[0] = joy_cmd[0]
        self.commands[1] = joy_cmd[1]
        self.commands[2] = joy_cmd[2]
        self.commands[3] = 0.0
        self.commands[4] = 0.0

        # Update control based on mode
        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()
        
        self.loop_count += 1

        # Apply control to MuJoCo
        for i in range(self.joint_num):
            data.ctrl[i] = (
                self.robot_cmd_Kp[i] * (self.robot_cmd_q[i] - self.robot_state_q[i]) + 
                self.robot_cmd_Kd[i] * (self.robot_cmd_dq[i] - self.robot_state_dq[i]) + 
                self.robot_cmd_tau[i]
            )


def load_callback(model=None, data=None):
    """Load callback for MuJoCo viewer."""
    mujoco.set_mjcb_control(None)

    model = mujoco.MjModel.from_xml_path(
        _MJCF_PATH.as_posix()
    )
    data = mujoco.MjData(model)

    # Reset to initial keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Set timestep
    ctrl_dt = 0.002  # 500Hz control frequency
    model.opt.timestep = ctrl_dt

    # Create controller
    controller = OnnxSolefootController()

    # Set control callback
    mujoco.set_mjcb_control(controller.get_control)

    return model, data


if __name__ == "__main__":
    print(f"Starting SF_TRON1A controller")
    viewer.launch(loader=load_callback)

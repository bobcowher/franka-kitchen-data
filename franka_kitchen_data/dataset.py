import os
from datetime import datetime

import numpy as np

# 1 is the original dataset, which has no version field; 2 adds finished_tasks, object_qpos and exclude_action.
SCHEMA_VERSION = 2


def done_labels(shard, task):
    """Done per frame for a prompt asking for task, from an opened shard of either version. Never reads object_qpos."""
    name = task.replace(" ", "_")
    if name == str(shard["task_name"]):
        return shard["done"].astype(bool)
    finished = shard["finished_tasks"] if "finished_tasks" in shard.files else []
    return np.full(len(shard["done"]), name in finished)


class DatasetShard():
    """One collection run's worth of VLA transitions, saved as its own file.

    Overrunning max_size raises rather than wrapping: a shard is a single
    episode, and a ring buffer would corrupt it instead of ageing out old data.

    Frames go in HWC, straight off the renderer. Do not record through
    ObsReshapeWrapper -- it hands out the reduced training size, not the
    render size the arena is built for.

    Measured over 104 microwave demos: 509 KiB per timestep, 59 steps per demo,
    3.2 GB. Demo length drives the total, not the task count.

    No next_state is stored. Behavior cloning never reads it, and within a shard
    it is an exact duplicate of the following row.
    """

    def __init__(self, max_size, image_size, n_actions, task_name,
                 task_description, finished_tasks=(), n_joints=9, n_objects=21):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.task_name = task_name
        self.task_description = task_description
        self.finished_tasks = [task.replace(" ", "_") for task in finished_tasks]
        self.object_qpos_memory = np.zeros((self.mem_size, n_objects), dtype=np.float32)
        self.exclude_action_memory = np.zeros(self.mem_size, dtype=bool)
        self.camera_scene_memory = np.zeros(
            (self.mem_size, image_size, image_size, 3), dtype=np.uint8)
        self.joint_pos_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.joint_vel_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def __len__(self):
        return self.mem_ctr

    def store_transition(self, state, action, reward, done, object_qpos, exclude_action=False):
        if self.mem_ctr >= self.mem_size:
            raise ValueError(
                f"shard is full at {self.mem_size} steps. Raise max_size to match "
                f"max_episode_steps, or the episode is running longer than expected.")

        index = self.mem_ctr

        self.camera_scene_memory[index] = state["camera_scene"]
        self.joint_pos_memory[index] = state["joint_pos"]
        self.joint_vel_memory[index] = state["joint_vel"]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.object_qpos_memory[index] = object_qpos
        self.exclude_action_memory[index] = exclude_action

        self.mem_ctr += 1

    def save_to_csv(self):

        date_formatted = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        directory = f'dataset/{self.task_name}'
        filename = f'{directory}/shard_{date_formatted}.npz'

        os.makedirs(directory, exist_ok=True)

        if os.path.exists(filename):
            raise FileExistsError(
                f"{filename} already exists. Shard names are second-resolution, "
                f"so this shard would overwrite one saved in the same second.")

        # savez_compressed leaves an invalid zip for the ~1s it runs, so write to
        # a temp name and os.replace it into place. A reader then sees the whole
        # shard or no file, which makes it safe to load a directory while a
        # collection run is still appending to it.
        tmp = f'{filename}.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh,
                     schema_version=SCHEMA_VERSION,
                     finished_tasks=np.array(self.finished_tasks, dtype=str),
                     object_qpos=self.object_qpos_memory[:self.mem_ctr],
                     exclude_action=self.exclude_action_memory[:self.mem_ctr],
                     task_name=self.task_name,
                     task_description=self.task_description,
                     camera_scene=self.camera_scene_memory[:self.mem_ctr],
                     joint_pos=self.joint_pos_memory[:self.mem_ctr],
                     joint_vel=self.joint_vel_memory[:self.mem_ctr],
                     action=self.action_memory[:self.mem_ctr],
                     reward=self.reward_memory[:self.mem_ctr],
                     done=self.terminal_memory[:self.mem_ctr])
        os.replace(tmp, filename)

        print("-" * 20)
        print(f"Saved {filename} ({self.mem_ctr} steps, "
              f"{os.path.getsize(filename) / 1e6:.1f} MB)")
        print("-" * 20)

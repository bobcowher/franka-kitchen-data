"""Record demos with the controller. Usage: python scripts/human_control.py <task> [--max-finished N]

A close task (e.g. "close slide cabinet") starts with its object open and ends once it is shut."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import BONUS_THRESH, OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES
import numpy as np
import pygame

from franka_kitchen_data import frames
from franka_kitchen_data.dataset import DatasetShard
from franka_kitchen_data.gym_robotics_custom import (FinishedTasksWrapper, HeldSetpointWrapper, RandomStartWrapper,
                                                     VLAObservationWrapper)
from franka_kitchen_data.tasks import CLOSE_TASKS, DESCRIPTIONS


def make_env(task, render_mode="human", max_finished=2, poses_path="start_poses.yaml", max_episode_steps=700):
    env_task = CLOSE_TASKS.get(task, task)
    # With both off, the env keeps running after success, so the demo ends when the operator says so.
    env = gym.make("FrankaKitchen-v1", max_episode_steps=max_episode_steps, tasks_to_complete=[env_task],
                   render_mode=render_mode, terminate_on_tasks_completed=False, remove_task_when_completed=False)
    env = HeldSetpointWrapper(env)
    env = RandomStartWrapper(env, poses_path=poses_path)
    env = FinishedTasksWrapper(env, env_task, max_finished=max_finished, start_open=task in CLOSE_TASKS)
    return VLAObservationWrapper(env, image_size=frames.RENDER_SIZE)


class DemoRecorder:

    def __init__(self, env, controller, task, max_steps=500, done_steps=20, step_sleep=0.05, done_delay=6,
                 done_margin=0.1, close_band_fraction=0.2, close_margin=0.03):
        self.env = env
        self.controller = controller
        self.task = task
        self.max_steps = max_steps
        self.done_steps = done_steps  # frames recorded after done is marked, with the operator still driving
        # Frames between first sim success and marking done, so the object is well inside the sim's band.
        self.done_delay = done_delay
        # How far inside the band the object must be at done; the hinge door swings back about 0.05 on release.
        self.done_margin = done_margin
        self.step_sleep = step_sleep
        self.indices = OBS_ELEMENT_INDICES[CLOSE_TASKS.get(task, task)]
        if task in CLOSE_TASKS:
            # Shut is the object's start qpos; the band matches vla2's close eval (0.2 of the open span).
            self.target = np.array(env.unwrapped.robot_env.init_qpos[self.indices], dtype=float)
            self.band = close_band_fraction * np.linalg.norm(OBS_ELEMENT_GOALS[CLOSE_TASKS[task]] - self.target)
            self.done_margin = close_margin
        else:
            self.target = OBS_ELEMENT_GOALS[task]
            self.band = BONUS_THRESH  # the env's own success test

    def record(self):
        """One episode. Returns (shard or None, whether to keep going)."""
        env, controller = self.env, self.controller
        self.state, _ = env.reset()
        controller.reset()
        finished = env.get_wrapper_attr("finished")
        print(f"\nTask: {DESCRIPTIONS[self.task]}. Already finished: {', '.join(finished) or 'nothing'}.")
        print("Ends by itself once the sim counts the task complete. Plus ends it early, d-pad starts and ends a fumble, Star discards, Minus quits.")

        self.memory = DatasetShard(max_size=self.max_steps + self.done_steps + 1,
                                   image_size=frames.RENDER_SIZE, n_actions=env.action_space.shape[0],
                                   task_name=self.task.replace(" ", "_"), task_description=DESCRIPTIONS[self.task],
                                   finished_tasks=finished)

        fumbling = miss_held = False
        success_frame = None
        while self.memory.mem_ctr < self.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, False

            if controller.quit_pressed():
                print("Quitting. This episode is not saved.")
                return None, False

            if controller.abort_pressed():
                print("Aborted. Discarding episode.")
                controller.wait_for_release()
                controller.buzz(2)
                return None, True

            if controller.done_pressed() and self.memory.mem_ctr > 0:
                controller.wait_for_release()
                return self._finish(), True

            miss = controller.miss_pressed()
            if miss and not miss_held:
                fumbling = not fumbling
                print(f"Fumble {'started' if fumbling else 'ended'}.")
            miss_held = miss

            action = controller.get_action()
            # Letting go of the pad right after success mustn't stall the countdown.
            if action is None and success_frame is not None:
                action = np.zeros(env.action_space.shape[0])
            if action is not None:
                complete = self._step(action, done=False, exclude_action=fumbling) > 0
                print(f"Episode step: {self.memory.mem_ctr} Complete: {complete}")
                if complete and success_frame is None:
                    success_frame = self.memory.mem_ctr
                if (success_frame is not None and self.memory.mem_ctr - success_frame >= self.done_delay
                        and self._goal_distance() <= self.band - self.done_margin):
                    self.memory.done_source = "sim"
                    return self._finish(), True
            time.sleep(self.step_sleep)

        print(f"No success or done press in {self.max_steps} steps. Discarding episode.")
        controller.buzz(2)
        return None, True

    def _goal_distance(self):
        return np.linalg.norm(self.env.unwrapped.data.qpos[self.indices] - self.target)

    def _step(self, action, done, exclude_action=False):
        # Taken before stepping, so qpos describes the same moment as self.state's frame.
        qpos = self.env.unwrapped.data.qpos.copy()
        next_state, _, _, _, _ = self.env.step(action)
        reward = float(self._goal_distance() < self.band)  # the env's own reward for open tasks
        self.memory.store_transition(self.state, action, reward, done, qpos, exclude_action)
        self.state = next_state
        return reward

    def _finish(self):
        """Once done is marked: keep recording the operator for done_steps frames, all labelled done.

        Whatever the operator does after finishing gives the done head positives with
        the arm in many poses, not one held pose. sim_complete notes whether the task
        was still complete at the end."""
        print(f"Done marked. Keep driving: {self.done_steps} more steps are recorded as done.")
        for _ in range(self.done_steps):
            pygame.event.pump()
            action = self.controller.get_action()
            if action is None:
                action = np.zeros(self.env.action_space.shape[0])  # pad released: hold still
            reward = self._step(action, done=True)
            time.sleep(self.step_sleep)
        self.sim_complete = reward > 0
        return self.memory


if __name__ == "__main__":
    from franka_kitchen_data.controller import Controller

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=lambda name: name.replace("_", " "), choices=list(DESCRIPTIONS))
    parser.add_argument("--max-finished", type=int, default=2, help="0 records a fresh scene")
    args = parser.parse_args()

    env = make_env(args.task, max_finished=args.max_finished)
    controller = Controller()
    recorder = DemoRecorder(env, controller, args.task)

    running = True
    while running:
        memory, running = recorder.record()
        if memory is not None:
            if controller.keep_demo(recorder.sim_complete):
                memory.save_to_csv()
                directory = f"dataset/{args.task.replace(' ', '_')}"
                shards = len([f for f in os.listdir(directory) if f.endswith(".npz")])
                print(f"{shards} demos recorded for {args.task}.")
                controller.buzz()
            else:
                print("Discarded.")
                controller.buzz(2)

    env.close()

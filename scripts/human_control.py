"""Record demos with the controller. Usage: python scripts/human_control.py <task> [--max-finished N]"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
import numpy as np
import pygame

from franka_kitchen_data import frames
from franka_kitchen_data.dataset import DatasetShard
from franka_kitchen_data.gym_robotics_custom import (ArmHomeWrapper, FinishedTasksWrapper, HeldSetpointWrapper,
                                                     RandomStartWrapper, VLAObservationWrapper)
from franka_kitchen_data.tasks import TASKS


def make_env(task, render_mode="human", max_finished=2, poses_path="start_poses.yaml", max_episode_steps=700):
    # With both off, the env keeps running after success, so the demo ends when the operator says so.
    env = gym.make("FrankaKitchen-v1", max_episode_steps=max_episode_steps, tasks_to_complete=[task],
                   render_mode=render_mode, terminate_on_tasks_completed=False, remove_task_when_completed=False)
    env = HeldSetpointWrapper(env)
    env = RandomStartWrapper(env, poses_path=poses_path)
    env = FinishedTasksWrapper(env, task, max_finished=max_finished)
    env = VLAObservationWrapper(env, image_size=frames.RENDER_SIZE)
    return ArmHomeWrapper(env)


class DemoRecorder:

    def __init__(self, env, controller, task, max_steps=500, hold_steps=20, step_sleep=0.05):
        self.env = env
        self.controller = controller
        self.task = task
        self.max_steps = max_steps
        self.hold_steps = hold_steps
        self.step_sleep = step_sleep

    def record(self):
        """One episode. Returns (shard or None, whether to keep going)."""
        env, controller = self.env, self.controller
        self.state, _ = env.reset()
        controller.reset()
        finished = env.get_wrapper_attr("finished")
        print(f"\nTask: {TASKS[self.task]}. Already finished: {', '.join(finished) or 'nothing'}.")
        print("Plus marks done, d-pad starts and ends a fumble, Star discards, Minus quits.")

        self.memory = DatasetShard(max_size=self.max_steps + self.hold_steps + env.max_steps + 1,
                                   image_size=frames.RENDER_SIZE, n_actions=env.action_space.shape[0],
                                   task_name=self.task.replace(" ", "_"), task_description=TASKS[self.task],
                                   finished_tasks=finished)

        fumbling = miss_held = False
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
            if action is not None:
                reward = self._step(action, done=False, exclude_action=fumbling)
                print(f"Episode step: {self.memory.mem_ctr} Reward: {reward}")
            time.sleep(self.step_sleep)

        print(f"No done press in {self.max_steps} steps. Discarding episode.")
        return None, True

    def _step(self, action, done, exclude_action=False):
        next_state, reward, _, _, _ = self.env.step(action)
        object_qpos = self.env.unwrapped.data.qpos[9:]
        self.memory.store_transition(self.state, action, reward, done, object_qpos, exclude_action)
        self.state = next_state
        return reward

    def _finish(self):
        """After a done press: hold still, go home, and report anything the sim says looks wrong."""
        print("Done marked. Holding, then going home.")
        hold_action = np.zeros(self.env.action_space.shape[0])
        for _ in range(self.hold_steps):
            reward = self._step(hold_action, done=True)
        if reward <= 0:
            print("Warning: the sim doesn't count this task as complete.")

        for _ in range(self.env.max_steps):
            action = self.env.home_action()
            if action is None:
                break
            reward = self._step(action, done=True)
        if reward <= 0:
            print("Warning: the sim doesn't count this task as complete after going home.")
        return self.memory


if __name__ == "__main__":
    from franka_kitchen_data.controller import Controller

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=lambda name: name.replace("_", " "), choices=list(TASKS))
    parser.add_argument("--max-finished", type=int, default=2, help="0 records a fresh scene")
    args = parser.parse_args()

    env = make_env(args.task, max_finished=args.max_finished)
    controller = Controller()
    recorder = DemoRecorder(env, controller, args.task)

    running = True
    while running:
        memory, running = recorder.record()
        if memory is not None:
            if controller.keep_demo():
                memory.save_to_csv()
                directory = f"dataset/{args.task.replace(' ', '_')}"
                shards = len([f for f in os.listdir(directory) if f.endswith(".npz")])
                print(f"{shards} demos recorded for {args.task}.")
            else:
                print("Discarded.")

    env.close()

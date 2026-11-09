# franka-kitchen-data

The FrankaKitchen demo collector and the code that defines its data. Projects consume the data; they don't own it.

## Install

    pip install -e ".[collect]"   # to record, from a clone
    pip install "franka-kitchen-data @ git+https://github.com/bobcowher/franka-kitchen-data.git@<full commit hash>"   # in a consumer

The second line is what goes in a consumer's `requirements.txt`. Pin the full commit hash; pip warns on a short one.

## Recording

From the repo root, where `dataset` is a symlink to `/data/datasets/franka-kitchen-v2`:

    python scripts/human_control.py "top burner"                     # other tasks start finished
    python scripts/human_control.py "top burner" --max-finished 0    # fresh scene

| Pad | Does |
|---|---|
| Plus | Marks the task done and ends the demo |
| D-pad | Starts or ends a fumble. Its frames stay out of the action loss |
| B / X at the end | Keep / discard |
| Star | Discards the episode and starts the next one |
| Minus | Quits without saving the current episode |

### One demo

1. Do the task. Sim success does not end the demo, so keep going until the task looks finished.
2. Press Plus. Your press is the done label, so press when you would tell the robot to stop.
3. Let go. The collector records 20 frames holding still where the task finished. The trip home is not recorded, because the done signal has to come before the arm leaves, not from the arm being home.
4. Wait for a long buzz, then press B to save or X to throw it away. A press before the buzz is ignored.
5. The pad buzzes once when a demo is saved and twice when one is discarded, then the next episode starts. The arm usually looks the same either way, so go by the buzz.

Good to know:

- A warning that the sim doesn't count the task as complete means you pressed before it passed the sim's threshold. Discard the demo with X unless you meant to keep it.
- If you make a mistake but can recover, press the d-pad, recover, and press the d-pad again. Star is for an episode that isn't worth saving.
- Steps only count while the pad is moving the arm. After 500 steps with no Plus press, the episode is discarded.
- Minus never saves. Finish with B first if you want the demo.

### Sending the data

Push new shards to lab with `scripts/sync_data.sh`.

## Data

- `/data/datasets/franka-kitchen-bc`: version 1, frozen. Called `farama-kitchen-bc/dataset` on lab.
- `/data/datasets/franka-kitchen-v2`: version 2.

One shard is one `.npz` episode:

| Field | Version | Meaning |
|---|---|---|
| `task_name`, `task_description` | 1, 2 | The commanded task |
| `camera_scene`, `joint_pos`, `joint_vel`, `action`, `reward` | 1, 2 | Per frame |
| `done` | 1, 2 | Version 1: the env's success on the last frame. Version 2: true from the operator's press through the hold. Shards saved before the hold-only change (2026-11-08) also have the trip home marked true |
| `schema_version` | 2 | Absent in version 1 |
| `finished_tasks` | 2 | Tasks already finished at the start |
| `exclude_action` | 2 | Per frame: inside a marked fumble |
| `qpos` | 2 | Per frame: the full sim state at that frame (arm then objects), for auditing, sim scoring and re-rendering from another camera |

`dataset.done_labels(shard, task)` gives per-frame labels for a prompt asking for any task.

Training labels must not read `qpos`, because a real robot has no equivalent. Frames from the done press onward and `exclude_action` frames should stay out of the action loss.

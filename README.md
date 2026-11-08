# franka-kitchen-data

The FrankaKitchen demo collector and the code that defines its data. Projects consume the data; they don't own it.

## Install

    pip install -e ".[collect]"   # to record, from a clone
    pip install "franka-kitchen-data @ git+ssh://git@github.com/bobcowher/franka-kitchen-data.git@<commit>"   # in a consumer

The second line is what goes in a consumer's `requirements.txt`. Pin the commit.

## Recording

From the repo root, where `dataset` is a symlink to `/data/datasets/franka-kitchen-v2`:

    python scripts/human_control.py "top burner"                     # other tasks start finished
    python scripts/human_control.py "top burner" --max-finished 0    # fresh scene

| Pad | Does |
|---|---|
| Plus | Marks the task done. The collector then holds still, goes home, and asks whether to keep the demo |
| D-pad | Starts or ends a fumble. Its frames stay out of the action loss |
| B / X at the end | Keep / discard |
| Star | Discards the episode |
| Minus | Quits |

Push new shards to lab with `scripts/sync_data.sh`.

## Data

- `/data/datasets/franka-kitchen-bc`: version 1, frozen. Called `farama-kitchen-bc/dataset` on lab.
- `/data/datasets/franka-kitchen-v2`: version 2.

One shard is one `.npz` episode:

| Field | Version | Meaning |
|---|---|---|
| `task_name`, `task_description` | 1, 2 | The commanded task |
| `camera_scene`, `joint_pos`, `joint_vel`, `action`, `reward` | 1, 2 | Per frame |
| `done` | 1, 2 | Version 1: the env's success on the last frame. Version 2: true from the operator's press onward |
| `schema_version` | 2 | Absent in version 1 |
| `finished_tasks` | 2 | Tasks already finished at the start |
| `exclude_action` | 2 | Per frame: inside a marked fumble |
| `qpos` | 2 | Per frame: the full sim state at that frame (arm then objects), for auditing, sim scoring and re-rendering from another camera |

`dataset.done_labels(shard, task)` gives per-frame labels for a prompt asking for any task.

Training labels must not read `qpos`, because a real robot has no equivalent. Frames from the done press onward and `exclude_action` frames should stay out of the action loss.

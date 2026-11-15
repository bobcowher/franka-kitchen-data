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

An open hinge cabinet gets in the way of the other tasks, so it's only already open in about 5% of their demos.

| Pad | Does |
|---|---|
| Plus | Ends the demo early, marking it done by hand |
| D-pad | Starts or ends a fumble. Its frames stay out of the action loss |
| B / X at the end | Keep / discard |
| Star | Discards the episode and starts the next one |
| Minus | Quits without saving the current episode |

### One demo

1. Do the task. The demo ends by itself once the sim has counted the task complete for 6 frames and the object is at least 0.1 inside the sim's 0.3 band. Keep pushing past the point where it first counts. The hinge door swings back about 0.05 when you let go.
2. Keep driving for 20 more steps, all labelled done. Do what you'd naturally do after finishing: ease off, back away a little, or let go of the pad (which holds still). The variety teaches the done head that the task is finished whatever the arm is doing. Leave the object alone; three buzzes at the end mean it left the band.
3. Wait for the buzz, then press B to save or X to throw it away. A press before the buzz is ignored. **One long buzz** means all is well. **Three long buzzes** mean the sim doesn't count the task as complete, so X is probably right.
4. The pad buzzes once when a demo is saved and twice when one is discarded, then the next episode starts. The arm usually looks the same either way, so go by the buzz.

Good to know:

- If you make a mistake but can recover, press the d-pad, recover, and press the d-pad again. Star is for an episode that isn't worth saving.
- Steps only count while the pad is moving the arm. After 500 steps without success or a Plus press, the episode is discarded.
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
| `done` | 1, 2 | Version 1: the env's success on the last frame. Version 2: true from the moment done is marked through the 20 steps after it |
| `schema_version` | 2 | Absent in version 1 |
| `done_source` | 2 | `sim`: at least 6 frames after first success, with the object at least 0.1 inside the band (from 2026-11-09). `press`: a Plus press; shards without this field are also presses. `sim_trimmed`: a press demo that ended outside the sim's band, cut back to where it left the band, with no hold. Originals are in `/data/datasets/franka-kitchen-v2-originals` |
| `finished_tasks` | 2 | Tasks already finished at the start |
| `exclude_action` | 2 | Per frame: inside a marked fumble |
| `qpos` | 2 | Per frame: the full sim state at that frame (arm then objects), for auditing, sim scoring and re-rendering from another camera |

`dataset.done_labels(shard, task)` gives per-frame labels for a prompt asking for any task.

Sim done labels are a sim-only choice: a real robot has no equivalent, so its demos will use the Plus press. Training code must not read `qpos`. Done frames and `exclude_action` frames should stay out of the action loss.

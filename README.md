# franka-kitchen-data

The FrankaKitchen demo collector and the code that defines its data. Projects consume the data; they don't own it.

The files here started as unmodified copies from `franka-kitchen-bc` (collector, controller, start poses) and `franka-kitchen-vla2` (wrappers, frames, tasks, dataset).

## Collecting

    python scripts/human_control.py "<task>"

Shards are written to `dataset/<task>/` under the working directory.

## Data

`/data/datasets/franka-kitchen-bc` is the original dataset: frozen, never modified.

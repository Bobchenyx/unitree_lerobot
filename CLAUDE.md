# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A bridge between Unitree humanoid robots (G1, Z1) and the HuggingFace [LeRobot](https://github.com/huggingface/lerobot) framework. It does three things:

1. **Convert** Unitree-recorded JSON datasets (from `avp_teleoperate`) into LeRobot dataset format and optionally push to HF Hub.
2. **Evaluate** trained LeRobot policies on a real Unitree robot (or in `unitree_sim_isaaclab`).
3. **Replay** datasets back onto the robot for validation.

Training itself is done with stock LeRobot scripts inside the submodule — this repo does not implement policies.

## Branch status: `v0.3.3-convert` is conversion-only

This branch pins the `lerobot` submodule to **v0.3.3** (`b883328`) specifically to keep the JSON → LeRobot conversion path working against the older lerobot API. **Only conversion is guaranteed here.** The eval / replay scripts (`eval_g1.py`, `eval_g1_sim.py`, `eval_g1_dataset.py`, `replay_robot.py`) and the matching sections of `README.md` target the v0.4.x API and will not run as-is on this branch — they import preprocessor/postprocessor symbols and episode-index fields that don't exist in v0.3.3 (see the v0.3.3 caveat in the next section). For v0.4.x integration use `main` / `NEU-dev`. The eval/replay commands later in this file are kept for architectural context but should not be invoked from this branch without porting.

## Repository layout

- `unitree_lerobot/lerobot/` — **git submodule** pinned to HuggingFace `lerobot` tag `v0.3.3` (commit `b883328`). Treat as third-party; don't edit unless deliberately patching. Init with `git submodule update --init --recursive`. Note: v0.3.3 policies do their own normalization internally — there is no separate preprocessor/postprocessor pipeline (those `lerobot.processor.PolicyAction` / `PolicyProcessorPipeline` / `make_pre_post_processors` / `rename_stats` symbols are v0.4.x-only, and their absence is why this branch's eval scripts don't run). Episode boundaries are exposed via `dataset.episode_data_index["from"|"to"]`, not `dataset.meta.episodes["dataset_from_index"|"dataset_to_index"]`.
- `unitree_lerobot/utils/` — data conversion. Four scripts live here:
  - `convert_unitree_json_to_lerobot.py` — main JSON → LeRobot pipeline; writes to `HF_LEROBOT_HOME / <repo_id>`; `--repo-id` required; image shape hardcoded to `(480, 640, 3)`.
  - `convert_unitree_json_to_lerobot_local.py` — same pipeline but adds `--root <path>` to write to an arbitrary directory, makes `--repo-id` optional (defaults to `local/<raw_dir_basename>`, required only with `--push_to_hub`), and auto-detects camera image shape from the first sample (so non-480×640 datasets work). This is what the working-tree `convert.sh` actually invokes.
  - `convert_unitree_json_to_h5.py` / `convert_lerobot_to_h5.py` — round-trip helpers between Unitree JSON, LeRobot, and HDF5; both are tyro CLIs.
  - `constants.py` holds `ROBOT_CONFIGS` (motor names, camera-to-image-key maps, JSON state/action keys per robot variant) and is the source of truth for what each `--robot_type` means.
- `unitree_lerobot/eval_robot/` — real-robot inference. `eval_g1.py` is the main entry; `eval_g1_sim.py` is the IsaacLab-sim variant; `eval_g1_dataset.py` runs a policy against a recorded dataset (no robot).
  - `make_robot.py` — `setup_robot_interface()` and `setup_image_client()` factory; central place where arm controller, IK solver, end-effector controller, optional mobile base, and image client are wired together via the `ARM_CONFIG` and `EE_CONFIG` dicts at the top of the file.
  - `robot_control/` — DDS-driven controllers per arm (`G1_29_ArmController`, `G1_23_ArmController`) and per end-effector (Dex3, Dex1, Inspire, Brainco). Communicates with the robot via `unitree_sdk2_python`.
  - `image_server/` — ZMQ image client/server pair; the server runs on the robot's onboard PC (see `avp_teleoperate` README).
  - `utils/utils.py` — `EvalRealConfig` (tyro-parsed config dataclass, the policy is loaded via LeRobot's `parser.wrap()` mechanism), `predict_action`, plus shared-memory cleanup helpers.
- `data_editor/` — standalone PyQt5 GUI (`data_editor_EN.py` / `data_editor_CN.py`) to trim/delete episodes from a raw Unitree dataset directory. Self-contained; does not import the rest of the package.
- `test/` — minimal smoke scripts (load a dataset, push a local dataset to hub). Not pytest tests.
- `convert.sh` / `sort_rename.sh` (repo root) — scratchpad/run-log shell scripts the user maintains for the conversion workflow. Old invocations are kept commented out as a history of what was run against which dataset; only the bottom (uncommented) line is live. Don't "tidy up" the commented blocks — they're intentional.

## Architecture notes worth knowing before editing

- **Robot variants are dictionary-driven, not class-hierarchical.** Adding a new robot/end-effector means appending to `ROBOT_CONFIGS` in `unitree_lerobot/utils/constants.py` (for conversion) and to `ARM_CONFIG` / `EE_CONFIG` at the top of `eval_robot/make_robot.py` (for inference). Keep the keys in sync — `--robot_type` strings (e.g. `Unitree_G1_Dex3`) come from `ROBOT_CONFIGS`, while `--arm` / `--ee` strings (e.g. `G1_29`, `dex3`) come from the eval-side dicts.
- **Shared memory is the IPC primitive.** `setup_robot_interface()` allocates `multiprocessing.Array`/`Value` blocks for end-effector state/action and hands them to the EE controller process. The eval loop reads/writes these arrays under `ee_shared_mem["lock"]`. Image frames flow through POSIX shared memory written by `ImageClient`. Always release them via `cleanup_resources()` in a `finally` — leaks persist across runs.
- **Action vector layout is positional, not keyed.** The eval loop slices `action_np[:arm_dof]`, `action_np[arm_dof:arm_dof+ee_dof]`, `action_np[arm_dof+ee_dof:arm_dof+2*ee_dof]`. The order is fixed: arm → left EE → right EE → (optional mobile base). `motors` order in `ROBOT_CONFIGS[...].motors` must match this layout, since it's also what the dataset is recorded with.
- **Config loading is two-pass.** `EvalRealConfig.__post_init__` re-parses CLI args to resolve `--policy.path` into a `PreTrainedConfig` via LeRobot's `parser`. Don't try to instantiate `EvalRealConfig` directly — go through `@parser.wrap()`.
- **JSON dataset shape.** `JsonDataset` (in `convert_unitree_json_to_lerobot.py`) expects `<raw_dir>/<task_name>/episode_XXXX/{data.json,colors/,depths/,audios/}`. It walks one level deep into `raw_dir`, so `--raw-dir` should point at the parent of the task folder, not the task folder itself. Episode directories must be sequentially named — run `sort_and_rename_folders.py` first if recording produced gaps.

## Common commands

Setup (matches README §1):

```bash
git submodule update --init --recursive
conda create -y -n unitree_lerobot python=3.10 && conda activate unitree_lerobot
conda install pinocchio ffmpeg=7.1.1 -c conda-forge
cd unitree_lerobot/lerobot && pip install -e . && cd ../..
pip install -e .
# For real-robot DDS:
# git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && cd unitree_sdk2_python && pip install -e .
# For data_editor: pip install PyQt5
```

Lint / format (pre-commit is configured; ruff is the source of truth, line length 120, target py310):

```bash
pre-commit run --all-files          # runs ruff-format, ruff, typos, gitleaks, bandit
ruff format . && ruff check --fix .  # quick local pass
```

Convert a Unitree JSON dataset to LeRobot format (the **primary workflow on this branch**):

```bash
# 1. Renumber episode_XXXX folders to be sequential (no gaps).
python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir $HOME/datasets/task_name

# 2a. Hub-style: writes to HF_LEROBOT_HOME/<repo_id>.
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets --repo-id user/task --robot_type Unitree_G1_Dex3 [--push_to_hub]

# 2b. Local-output variant (what convert.sh currently runs): writes to --root,
#     --repo-id optional, image shape auto-detected from first frame.
python unitree_lerobot/utils/convert_unitree_json_to_lerobot_local.py \
    --raw-dir /path/to/raw --robot_type Unitree_G1_Dex1_Sim --root /path/to/lerobot-out
```

Train (delegates to LeRobot inside the submodule — note: training on this branch uses the v0.3.3 API):

```bash
cd unitree_lerobot/lerobot
python src/lerobot/scripts/train.py --dataset.repo_id=... --policy.type=act|diffusion|pi0|pi0fast|smolvla|tdmpc|vqbet
```

Run a trained policy on the real G1 (⚠ **broken on this branch** — kept for context; use `main`/`NEU-dev` to actually run it):

```bash
# image_server must be running on the robot first (see avp_teleoperate README §3.1)
python unitree_lerobot/eval_robot/eval_g1.py \
    --policy.path=<path-to-pretrained_model> --repo_id=<dataset-repo> \
    --arm=G1_29 --ee=dex3 --frequency=30 --visualization=true
```

Replay a recorded episode on the real robot (⚠ **broken on this branch**, same reason):

```bash
python unitree_lerobot/eval_robot/replay_robot.py --repo_id=... --arm=G1_29 --ee=dex3 --episodes=0
```

Smoke-check a dataset loads (there is no pytest harness — these are scripts):

```bash
python test/test_load_dataset.py
```

## Safety notes specific to this repo

- `eval_g1.py` and `replay_robot.py` move a real robot. The eval loop blocks on `input("Enter 's' to start...")` for a reason — don't remove that prompt.
- `unitree_lerobot/lerobot/` is a submodule. Do not commit changes to files inside it as part of this repo without an explicit reason; submodule pointer bumps are a separate concern from upstream patches.
- The `*.json` gitattribute disables LFS/text filtering for JSON, so dataset JSONs (if ever committed) round-trip byte-exactly. Don't add JSON to `.gitignore`-adjacent rules without checking.

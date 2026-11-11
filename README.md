# MultiTalk

Code for *MultiTalk: Introspective and Extrospective Dialogue for Human-Environment-LLM
Alignment* (ICRA). Project website: <https://llm-multitalk.github.io/>

MultiTalk turns a natural-language instruction into a plan for a Franka FR3 arm. A Planner
LLM and an Analyzer LLM critique plans together, while the user, perception and a MuJoCo
simulator supply feedback from outside the LLMs.

> Real-robot execution is not included yet. Plans are saved as text after passing the
> simulator.

> **Reproducibility:** The experiments in the paper were run with an earlier version of
> GPT-4o that OpenAI has since deprecated. This code uses `gpt-4o`, which points to OpenAI's
> current GPT-4o version, so results may differ from those reported in the paper and are not
> exactly reproducible.

## Install

Requires Python 3.10. PyTorch is only needed for perception.

```bash
conda create -n multitalk python=3.10 pip
conda activate multitalk
export PYTHONNOUSERSITE=1            # keep ~/.local packages out of the environment

pip install -e .

# Optional: perception (adjust the CUDA version to your machine)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[perception]"
bash scripts/download_weights.sh
```

Add your OpenAI key to a `.env` file in the project root, which git ignores:

```bash
cp .env.example .env
```

## Usage

Generate a plan for a bundled scene:

```bash
multitalk plan --scene exp8/new_objects --task "Interchange the locations of two objects"
```

Or detect the scene from an RGB-D capture containing `rgb.png`, `depth.png`, `R_hand.npy`
and `p_hand.npy`:

```bash
multitalk plan --capture-dir path/to/capture --task "Give me something to eat"
```

Each run is saved in `runs/<timestamp>/`:

| File | Contents |
|---|---|
| `final_plan.txt` | The last plan the Analyzer accepted |
| `result.json` | `simulator_passed`, iterations and estimated cost |
| `planner_conversation.txt`, `analyzer_conversation.txt` | The full LLM dialogues |

`final_plan.txt` has passed the simulator only if `simulator_passed` is `true`. The command
exits with status 0 in that case and 1 otherwise.

Other commands:

```bash
multitalk simulate --scene exp8/new_objects --plan plan.txt --render
multitalk detect --capture-dir path/to/capture --output scene.json
```

When the Planner requests a rescan, MultiTalk waits for you to update the capture or scene
file, because moving the camera is not automated yet.

## Configuration

`configs/default.yaml` lists every setting. Use `multitalk --config my_config.yaml plan ...`.
The `checks` section turns the bounds check, Analyzer and Simulator on or off for ablated
runs.

## Layout

```text
src/multitalk/
├── pipeline.py      # The planning loop (Algorithm 1)
├── cli.py           # Command line interface
├── plan_text.py     # Plan and question text handling
├── config.py        # Settings and paths
├── types.py         # Scene, plan steps and results
├── llm/             # OpenAI and Ollama clients, Planner, Analyzer
├── sim/             # Scene builder, controller, simulator
├── perception/      # GroundingDINO + MobileSAM + depth
├── prompts/         # Prompt templates
└── assets/          # Robot and object models, camera calibration, saved scenes
```

## Third-party code and assets

See `third_party_licenses/`. The controller, simulation environment and robot models are
adapted from [FR3Env](https://github.com/BolunDai0216/FR3Env) (MIT). Object models come from
[YCB_sim](https://github.com/vikashplus/YCB_sim) (Apache-2.0), based on the
[YCB object set](https://www.ycbbenchmarks.com/).

## Citation

```bibtex
@inproceedings{devarakonda2025multitalk,
  title     = {MultiTalk: Introspective and Extrospective Dialogue for Human-Environment-LLM Alignment},
  author    = {Devarakonda, Venkata Naren and Kaypak, Ali Umut and Yuan, Shuaihang and
               Krishnamurthy, Prashanth and Fang, Yi and Khorrami, Farshad},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```

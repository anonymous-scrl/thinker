# Thinker: Learning to Plan and Act

This is the official repository for the paper titled "Thinker: Learning to Plan and Act". 

## Prerequisites

Ensure that Pytorch is installed (versions v2.0.0 and v1.13.0 have been tested).

## Installation

1. Update and install the necessary packages:

```bash
sudo apt-get update
sudo apt-get install python-opencv build-essential -y
pip install -r requirement.txt
```

2. Install the C++ version of Sokoban (skip this step if you're not running experiments on Sokoban):
```bash
cd csokoban
pip install -e .
```

3. Compile the Cython version of the wrapped environment:
```bash
cd thinker
python setup.py build_ext --inplace
```

## Usage
Run the following commands in the `thinker` directory:

Sokoban default run:

```bash
python train.py --env cSokoban-v0 --model_size_nn 1 --disable_frame_copy --discounting 0.97 --reward_clip -1
```

Raw MDP Sokoban default run:

```bash
python train.py --env cSokoban-v0 --model_size_nn 1 --disable_frame_copy --discounting 0.97 --reward_clip -1 --disable_model --unroll_length 20 --learning_rate 0.0003
```

The above default and the raw MDP run are used to generate the results of the Thinker-augmented MDP and the raw MDP in Figure 4 of the paper, respectively.

Atari default run (change the environment if needed):

```bash
python train.py --env BreakoutNoFrameskip-v4
```

Raw MDP Atari default run:

```bash
python train.py --env BreakoutNoFrameskip-v4 --disable_model --unroll_length 20 --learning_rate 0.0003
```

The above default and the raw MDP run used to generate the results of the Thinker-augmented MDP and the raw MDP in Figure 7 of the paper, respectively.

If you encounter errors related to Ray memory, try setting `--ray_mem -1` to allow Ray to allocate memory automatically.

The number of GPUs will be detected automatically. A single RTX3090 is sufficient for the Sokoban default run, while two RTX3090s are required for the Atari default run. The number of CPUs and GPUs allocated can be controlled with the `--ray_cpu` and `--ray_gpu` options, e.g., `--ray_cpu 16 --ray_gpu 1` limits usage to 16 CPUs and 1 GPU.

## Trained Agents

The trained agents used in the behavior analysis of Appendix G are available in: https://drive.google.com/file/d/1o--3m3bfdZdC7ssSnk0vF8KEFOkCKzOL

Run the following commands in the `thinker` directory to visualize a run of the trained agents (replace `base` with the path of the downloaded runs)

```bash
python visual.py --load_checkpoint base --output_path test
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
[Redacted]

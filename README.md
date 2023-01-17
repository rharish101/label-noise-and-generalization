# Label Noise, Schedulers and Generalization

This is a repository for my semester project (Research in Computer Science: 263-0600-00L) under the [Data Analytics group](http://da.inf.ethz.ch/) at ETH Zürich in the fall semester of 2021.
For a detailed explanation of the idea, the approach and the findings, please check the [project report](./report.pdf).

#### Advisors
* Antonio Orvieto
* Jonas Kohler

## Setup
[Poetry](https://python-poetry.org/) is used for conveniently installing and managing dependencies.
[pre-commit](https://pre-commit.com/) is used for managing hooks that run before each commit, to ensure code quality and run some basic tests.

1. *[Optional]* Create and activate a virtual environment with Python >= 3.8.5.

2. Install Poetry globally (recommended), or in a virtual environment.
    Please refer to [Poetry's installation guide](https://python-poetry.org/docs/#installation) for recommended installation options.

    You can use pip to install it:
    ```sh
    pip install poetry
    ```

3. Install all dependencies, including extra dependencies for development, with Poetry:
    ```sh
    poetry install
    ```

    To avoid installing development dependencies, run:
    ```sh
    poetry install --no-dev
    ```

    If you didn't create and activate a virtual environment in step 1, Poetry creates one for you and installs all dependencies there.
    To use this virtual environment, run:
    ```sh
    poetry shell
    ```

4. Install pre-commit hooks:
    ```sh
    pre-commit install
    ```

**NOTE:** You need to be inside the virtual environment where you installed the above dependencies every time you commit.
However, this is not required if you have installed pre-commit globally.

## Tasks

The optimizers are tested on multiple tasks.
Each task involves training a certain model in a certain manner (supervised, unsupervised, etc.) on a certain dataset.
Every task is given a task ID, which is used when running scripts.

The list of tasks implemented, along with their IDs, are:

Task | Task ID | Description
-- | -- | --
CIFAR-10 | `cifar` | A ResNet-18 on the CIFAR10 dataset.

## Scripts

All scripts use argparse to parse commandline arguments.
Each Python script takes the task ID as a positional argument.
To view the list of all positional and optional arguments for a script `script.py`, run:
```sh
./script.py --help
```

## Hyper-Parameters

### Configuration
Hyper-parameters can be specified through YAML configs.
For example, to specify a batch size of 32 and a learning rate of 0.001, use the following config:
```yaml
lr: 0.001
batch_size: 32
```

You can store configs in a directory named `configs` located in the root of this repository.
It has an entry in the [`.gitignore`](./.gitignore) file so that custom configs aren't picked up by git.

The available hyper-parameters, their documentation and default values are specified in the `Config` class in the file [`src/config.py`](./src/config.py).

**NOTE:** You do not need to mention every single hyper-parameter in a config.
In such a case, the missing ones will use their default values.

### Tuning
Support for tuning hyper-parameters for the optimizers is available in the training script.
It has the `-m` or the `--mode` flag to set the mode of operation.
This has the following values:
* `train`: This simply trains a model. This is the default mode.
* `tune`: This tunes the hyper-parameters using [Hyperopt](https://github.com/hyperopt/hyperopt).

Thus, to tune hyper-parameters for models on a certain task, run the training script as follows:
```sh
./train.py --mode tune
```

## Logs
Logs are stored with certain directory structures.
For training, this is:
```
project root
|_ root log directory
   |_ experiment name
      |_ timestamped run directory
```

For tuning, this is:
```
project root
|_ root log directory
   |_ experiment name
      |_ timestamped tuning run directory
         |_ training run 0 directory
         |_ training run 1 directory
         ...
```

The timestamp uses the ISO 8601 convention along with the local timezone.
The root log directory can be specified with the `--log-dir` argument.
By default, this is `logs`.

The sub-directory for each training run will contain:
* The latest checkpoint of the trained model, within the `checkpoints` sub-directory
* Training logs, as a file with the prefix `events.out.tfevents.`
* The hyper-parameter config (including defaults), as a YAML file named `hparams.yaml`

The sub-directory for a tuning run will contain:
* Sub-directories for each training run
* The best hyper-parameter config (including defaults), as a YAML file named `best-hparams.yaml`

## Miscellaneous Features

### Multi-GPU Training
For choosing which GPUs to train on, use the `-g` or the `--num-gpus` flag when running a script as follows:
```sh
./script.py --num-gpus 3
```

This selects three available GPUs for training.
By default, only one GPU is chosen.

### Mixed Precision Training
This implementation supports mixed-precision training, which is enabled by default.
To manually set the floating-point precision, use the `-p` or the `--precision` flag when running a script as follows:
```sh
./script.py --precision 32
```

Note that mixed-precision training will only provide significant speed-ups if your GPUs have special support for mixed-precision compute.

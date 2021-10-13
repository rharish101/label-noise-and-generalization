# Adaptive Generalization

This is a repository for my semester project (Research in Computer Science: 263-0600-00L) under the [Data Analytics group](http://da.inf.ethz.ch/) at ETH Zürich in the fall semester of 2021.

#### Advisors
* Antonio Orvieto
* Jonas Kohler

## Instructions

All Python scripts use argparse to parse commandline arguments.
To view the list of all positional and optional arguments for a script `script.py`, run:
```sh
./script.py --help
```

### Setup
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

### Hyper-Parameter Configuration
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

#### Multi-GPU Training
For choosing which GPUs to train on, use the `-g` or the `--num-gpus` flag when running a script as follows:
```sh
./script.py --num-gpus 3
```

This selects three available GPUs for training.
By default, only one GPU is chosen.

#### Mixed Precision Training
This implementation supports mixed-precision training, which is enabled by default.
To manually set the floating-point precision, use the `-p` or the `--precision` flag when running a script as follows:
```sh
./script.py --precision 32
```

Note that mixed-precision training will only provide significant speed-ups if your GPUs have special support for mixed-precision compute.

## Tasks
### CIFAR10
This uses a ResNet-18 to train on the CIFAR10 dataset.

Run the script `cifar10.py`:
```sh
./cifar10.py
```

Training logs are by default stored inside an ISO 8601 timestamped sub-directory, which is stored in the `cifar10` sub-directory of a parent directory (as given by the `--log-dir` argument).
By default, this parent directory is `logs`.
This contains:
* The latest checkpoint of the trained model, within the `checkpoints` sub-directory
* Training logs, as a file with the prefix `events.out.tfevents.`
* The hyper-parameter config (including defaults), as a YAML file named `hparams.yaml`

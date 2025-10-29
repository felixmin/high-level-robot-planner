# Comprehensive Hydra Framework Documentation

## Overview

Hydra is an open-source Python framework that simplifies the development of research and other complex applications by providing **dynamic hierarchical configuration management**. Its key distinguishing feature is the ability to compose configurations from multiple sources and override them through config files and command line arguments. The name "Hydra" comes from its ability to run multiple similar jobs—much like a mythological Hydra with multiple heads.[1][2]

## Core Features and Capabilities

### Configuration Composition and Management

Hydra operates on top of **OmegaConf**, a YAML-based hierarchical configuration system that supports merging configurations from multiple sources including files, CLI arguments, and environment variables. The framework allows you to:[3]

- Create hierarchical configurations composable from multiple sources[2]
- Specify or override configuration from the command line[2]
- Dynamically compose configurations through **config groups**[2]
- Use **structured configs** with Python dataclasses for type validation[4][5]

### Multi-Run Capabilities

One of Hydra's most powerful features is its ability to execute **parameter sweeps** using the `--multirun` or `-m` flag. This enables:[6][2]

- Running multiple jobs with different parameter combinations in a single command[2]
- Local parallel execution or distributed execution across clusters[7][6]
- Integration with job schedulers like SLURM through plugins[8][7]

### Working Directory Management

Hydra automatically creates **timestamped output directories** for each run, following the pattern `outputs/YYYY-mm-dd/HH-MM-SS`. This provides:[9][6]

- Automatic experiment tracking without additional effort[6]
- Preservation of configuration files, logs, and outputs for each run[9]
- Optional working directory changes via `hydra.job.chdir=True`[9]

## Installation and Basic Usage

### Installation
```bash
pip install hydra-core --upgrade
```

### Basic Example

**Configuration file (config.yaml):**
```yaml
db:
  driver: mysql
  user: omry
  pass: secret
```

**Application (my_app.py):**
```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

**Command line override:**
```bash
python my_app.py db.user=root db.pass=1234
```

## Advanced Configuration Patterns

### Config Groups and Composition

Config groups enable **modular configuration management** by organizing related configurations into directories. Each config group represents a set of mutually exclusive options:[10][11]

**Directory structure:**
```
├── conf
│   ├── config.yaml
│   ├── db
│   │   ├── mysql.yaml
│   │   └── postgresql.yaml
│   └── optimizer
│       ├── adam.yaml
│       └── sgd.yaml
```

**Main config (config.yaml):**
```yaml
defaults:
  - db: mysql
  - optimizer: adam
```

This allows switching between configurations using CLI overrides:
```bash
python my_app.py db=postgresql optimizer=sgd
```

### Package Directives and the @_here_ Annotation

Hydra uses **package directives** to control where configuration content is placed within the final config object. The `@_here_` annotation is particularly important for configuration placement:[12][13]

#### When to Use @_here_

The `@_here_` package keyword should be used when you want to **merge configuration content into the current package level** rather than creating a new nested structure. This is especially useful in the following scenarios:[1][2]

1. **Flattening nested configurations**: When you want to avoid creating additional nesting levels in your final configuration
2. **Merging related configurations**: When combining multiple config files that should appear at the same hierarchical level
3. **Inheritance patterns**: When extending base configurations while maintaining the same package structure

**Example using @_here_:**
```yaml
# config.yaml
defaults:
  - base_config@_here_: common
  - db: mysql

training:
  epochs: 100
```

In this example, the contents of `base_config/common.yaml` will be merged directly into the root level of the configuration, rather than being nested under a `base_config` key.[14][1]

#### Package Keywords Reference

- **`@_here_`**: Merges content into the current config group package[12]
- **`@_global_`**: Places content at the root level of the configuration[13][12]
- **`@_group_`**: Places content under the config group name (default behavior)[15][12]

### Structured Configs with Dataclasses

Structured configs provide **type safety and validation** using Python dataclasses:[5][16][4]

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class DatabaseConfig:
    driver: str
    host: str = "localhost"
    port: int = 3306

@dataclass
class Config:
    db: DatabaseConfig
    batch_size: int = 32

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
```

### Variable Interpolation

Hydra supports **variable interpolation** for dynamic configuration values:[17][18]

```yaml
optimizer:
  learning_rate: 0.001
  
model:
  layers: 3
  
training:
  lr: ${optimizer.learning_rate}  # Reference other config values
  model_name: "model_${model.layers}_layers"  # String interpolation
```

**Custom resolvers** can be registered for complex interpolations:[17]
```python
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)

# Usage in config: ${multiply:${batch_size},4}
```

## Multi-Run and Experiment Management

### Parameter Sweeps

Hydra excels at **hyperparameter optimization and experimentation**:[19][6]

```bash
# Single parameter sweep
python train.py --multirun learning_rate=0.001,0.01,0.1

# Multiple parameter combinations
python train.py --multirun optimizer=adam,sgd learning_rate=0.001,0.01 batch_size=16,32
```

This automatically runs all combinations (2 × 2 × 2 = 8 jobs in the second example).[6]

### Launcher Plugins

Hydra supports various **execution backends** through launcher plugins:[8][7]

- **Local launcher**: Sequential execution on local machine
- **Joblib launcher**: Parallel execution using joblib
- **Submitit launcher**: SLURM cluster execution[7][8]
- **Ray launcher**: Distributed execution on Ray clusters

**SLURM example configuration:**
```yaml
# hydra/launcher/slurm.yaml
defaults:
  - submitit_slurm

timeout_min: 120
gpus_per_node: 1
nodes: 1
tasks_per_node: 1
```

### Working Directory and Output Management

Hydra provides sophisticated **output organization**:[20][9]

- **Single runs**: `outputs/YYYY-mm-dd/HH-MM-SS/`
- **Multi-runs**: `multirun/YYYY-mm-dd/HH-MM-SS/0/`, `multirun/YYYY-mm-dd/HH-MM-SS/1/`, etc.

Each output directory contains:[9]
- `.hydra/config.yaml`: Final composed configuration
- `.hydra/hydra.yaml`: Hydra runtime configuration  
- `.hydra/overrides.yaml`: Command line overrides used
- Application logs and outputs

**Customizing output directories:**
```yaml
hydra:
  run:
    dir: ./experiments/${now:%Y-%m-%d}/${hydra.job.name}
  sweep:
    dir: ./sweeps/${now:%Y-%m-%d}/${hydra.job.name}
    subdir: ${hydra.job.num}
```

## Override Syntax and Command Line Usage

### Basic Override Patterns

Hydra provides a **comprehensive override syntax**:[21]

- **Modify existing**: `key=value`
- **Add new**: `+key=value` 
- **Delete**: `~key`
- **Empty value**: `key=`

### Complex Value Handling

**Quoted values** support complex data types:[21]
```bash
# Strings with spaces
python app.py 'message=hello world'

# Complex structures  
python app.py 'model={layers: 3, activation: relu}'

# Lists
python app.py 'learning_rates=[0.1, 0.01, 0.001]'
```

## Best Practices and Integration Patterns

### Experiment Organization

Effective Hydra usage involves **structured experiment management**:[22][19]

1. **Separate concerns**: Use config groups for different aspects (model, data, optimizer)
2. **Base configurations**: Create base configs that are extended by specific experiments
3. **Experiment configs**: Use dedicated experiment config groups for reproducible setups
4. **Parameter validation**: Employ structured configs for type safety

### Integration with ML Frameworks

Hydra integrates well with **popular ML frameworks**:[19]

- **Instantiation**: Use `hydra.utils.instantiate()` for object creation from configs
- **Logging integration**: Works with MLflow, Weights & Biases, TensorBoard
- **Distributed training**: Supports PyTorch distributed training through launcher plugins[4]

### Configuration Inheritance

**Extending configurations** enables code reuse:[23][24]
```yaml
# base/generic.yaml
batch_size: 128
num_workers: 8
dataset: ???

# data/cifar10.yaml  
defaults:
  - base/generic

dataset: cifar10
batch_size: 64  # Override base value
augment: true   # Add new parameter
```

## Platform Support and Ecosystem

### Supported Platforms

Hydra supports **cross-platform development**:[2]
- Linux, macOS, and Windows
- Python 3.6 - 3.11 (depending on Hydra version)

### Plugin Ecosystem

The Hydra ecosystem includes numerous **community plugins**:[4][8]
- Launcher plugins for various execution backends
- Sweeper plugins for hyperparameter optimization
- Integration plugins for ML platforms

### Tab Completion

Hydra provides **dynamic command-line tab completion** for enhanced developer experience:[2]
```bash
# Get completion installation command
python my_app.py --hydra-help

# Supports Bash, Zsh, and Fish shells
```

This comprehensive documentation covers Hydra's core functionality, advanced features, and practical usage patterns. The framework's strength lies in its ability to **separate configuration from code**, enable **reproducible experiments**, and provide **flexible parameter management** for complex applications, particularly in machine learning and research contexts.

[1](https://stackoverflow.com/questions/67715171/fb-hydra-how-to-get-inner-configurations-to-inherit-outer-configuration-fields)
[2](https://stackoverflow.com/questions/78572906/is-referring-to-default-values-in-another-file-allowed-in-hydra-config)
[3](https://deep-learning-blogs.vercel.app/blog/mlops-hydra-config)
[4](https://github.com/acherstyx/hydra-torchrun-launcher)
[5](https://www.gaohongnan.com/software_engineering/config_management/01-pydra.html)
[6](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b/)
[7](https://hydra.cc/docs/plugins/submitit_launcher/)
[8](https://www.aidanscannell.com/notes/hpc-cluster/hydra/)
[9](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/)
[10](https://hydra.cc/docs/1.0/tutorials/basic/your_first_app/config_groups/)
[11](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/)
[12](https://hydra.cc/docs/advanced/overriding_packages/)
[13](https://github.com/facebookresearch/hydra/issues/1913)
[14](https://stackoverflow.com/questions/68984309/overriding-the-package-specified-in-default-list-from-cli)
[15](https://hydra.cc/docs/1.0/advanced/overriding_packages/)
[16](https://hydra.cc/docs/tutorials/structured_config/intro/)
[17](https://omegaconf.readthedocs.io/en/2.1_branch/custom_resolvers.html)
[18](https://sscardapane.it/tutorials/hydra-tutorial/)
[19](https://docs.jarvislabs.ai/blog/ml-tracking)
[20](https://hydra.cc/docs/configure_hydra/workdir/)
[21](https://hydra.cc/docs/advanced/override_grammar/basic/)
[22](https://towardsdatascience.com/keep-track-of-your-experiments-with-hydra-b29937a99fc9/)
[23](https://hydra.cc/docs/patterns/extending_configs/)
[24](https://www.creatis.insa-lyon.fr/newsletter/2023_01_04/pdfs/club_dev_hydra.pdf)
[25](https://stackoverflow.com/questions/tagged/fb-hydra)
[26](https://github.com/facebookresearch/hydra/issues/1803)
[27](https://aaronyoung5.github.io/hydra-config/reference/api/hydra_config/index.html)
[28](https://dvc.org/doc/user-guide/experiment-management/hydra-composition)
[29](https://clear.ml/docs/latest/docs/integrations/hydra/)
[30](https://github.com/facebookresearch/hydra/discussions/2126)
[31](https://www.youtube.com/watch?v=t9hwWxBnU0o)
[32](https://www.reddit.com/r/MachineLearning/comments/1n8lvz5/d_how_do_you_read_code_with_hydra/)
[33](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/)
[34](https://maze-rl.readthedocs.io/en/latest/concepts_and_structure/hydra/custom_config.html)
[35](https://github.com/facebookresearch/hydra/discussions/2836)
[36](https://ohdsi.github.io/Hydra/articles/WritingHydraConfigs.html)
[37](https://www.kdnuggets.com/2023/03/hydra-configs-deep-learning-experiments.html)
[38](https://github.com/facebookresearch/hydra/issues/1622)
[39](https://hydra.cc/docs/advanced/defaults_list/)
[40](https://schnetpack.readthedocs.io/en/latest/userguide/configs.html)
[41](https://towardsdatascience.com/configuration-management-for-model-training-experiments-using-pydantic-and-hydra-d14a6ae84c13/)
[42](https://stackoverflow.com/questions/73755847/how-to-use-a-config-group-multiple-times-while-overriding-each-instance)
[43](https://hydra.cc/docs/advanced/terminology/)
[44](https://hydra.cc/docs/advanced/search_path/)
[45](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-15/hydra-environment-variables.html)
[46](https://prace-ri.eu/wp-content/uploads/Best-Practice-Guide_Hydra.pdf)
[47](https://pkg.go.dev/github.com/comstud/hydra/config)
[48](https://hydra.cc/docs/0.11/tutorial/composition/)
[49](https://stackoverflow.com/questions/79020464/using-hydra-to-select-multiple-structured-configs)
[50](https://github.com/pooya-mohammadi/hydra_examples)
[51](https://mit-ll-responsible-ai.github.io/hydra-zen/generated/hydra_zen.hydrated_dataclass.html)
[52](https://imperialcollegelondon.github.io/ReCoDE-DeepLearning-Best-Practices/learning/Learning_about_hydra/)
[53](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
[54](https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html)
[55](https://hydra.cc/docs/advanced/instantiate_objects/config_files/)
[56](https://github.com/facebookresearch/hydra/discussions/2526)
[57](https://maze-rl.readthedocs.io/en/latest/concepts_and_structure/hydra/advanced.html)
[58](https://dsip.pages.fbk.eu/dsip-docs/machine-learning/slurm-guide/)
[59](https://hydra.cc/docs/advanced/instantiate_objects/structured_config/)
[60](https://github.com/FlorianWilhelm/hydra-example-project)
[61](https://github.com/facebookresearch/hydra/issues/2800)
[62](https://maze-rl.readthedocs.io/en/latest/concepts_and_structure/hydra/overview.html)
[63](https://neptune.ai/blog/experiment-management)
[64](https://stackoverflow.com/questions/73007303/what-is-the-correct-way-of-accessing-hydras-current-output-directory)
[65](https://github.com/facebookresearch/hydra/issues/2414)
[66](https://hydra.cc/docs/patterns/specializing_config/)
[67](https://hydra.cc/docs/patterns/configuring_experiments/)
[68](https://forums.developer.nvidia.com/t/save-outputs-to-a-different-folder/242963)
[69](https://stackoverflow.com/questions/78242414/why-does-the-value-interpolation-not-work-when-using-python-hydra-library)
[70](https://github.com/facebookresearch/hydra/issues/910)
[71](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
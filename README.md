# RLogist

Fast Observation Strategy on Whole-slide Images with Deep Reinforcement Learning

## Fully Automated Run

### Data Preparation

The data used for training and testing are expected to be organized as follows: 

- Digitized WSI data in well known standard formats (.svs, .ndpi, .tiff etc.) are stored under a folder named DATA_DIRECTORY 

```
DATA_DIRECTORY/
	├── slide_1.tif
	├── slide_2.tif
	└── ...
```

- WSI labels are recorded in a .CSV file: LABEL_LIST.csv

  | WSI_path                        | label |
  | ------------------------------- | ----- |
  | CAMELYON16/train/tumor_001.tif  | 1     |
  | CAMELYON16/train/normal_001.tif | 0     |

### Automated Run

```shell
python main.py --source DATA_DIRECTORY --label_list LABEL_LIST.csv
```

The script automatically reads the dataset and the corresponding labels for training. 

The segmentation and patching settings can be configured in `create_patches.py`, training parameters for RLogist can be configured in `main.py` (RL algorithm-specific hyper-parameters in source file like `ppo.py`)

### Evaluation

```shell
python eval_model.py --input_dir DATA_DIRECTORY --config_file CONFIG.yaml
```

The script automatically loads the model and reads the dataset and the corresponding labels for evaluation. 

## Components 

Follow the Guidance in corresponding directories:

### RL_env Test

```shell
python WSI_observation_env.py
```

### CLAM classifier Pretrain

```shell
python train_CLAM_model.py
```


## Table of Contents
- [License Notice](#license-notice)
- [Usage](#usage)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Citation](#citation)

## License Notice

This repository includes code that enables the user to load models from various sources with different licenses:

- **MIT Licensed Code**: [This repository Transferability-SSL-Wound-Recognition](https://github.com/julien-marteen-akay/Transferability-SSL-Wound-Recognition) is licensed under the [MIT License](https://github.com/julien-marteen-akay/Transferability-SSL-Wound-Recognition/blob/main/LICENSE).

- **CC BY-NC 4.0 Licensed Models**: Some models that can be downloaded and used through the code in this repository are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). These models are for non-commercial use only. See details below.

### Attribution for CC BY-NC 4.0 Licensed Models

All models from the repository accompanying the [VICRegL paper](https://arxiv.org/abs/2210.01571) are licensed under CC BY-NC 4.0.
- Original Authors: Adrien Bardes and Jean Ponce and Yann LeCun
- Source: https://github.com/facebookresearch/VICRegL
- License: https://github.com/facebookresearch/VICRegL/blob/main/LICENSE
- **<u>How to avoid using these models</u>:**
  - When starting the training process, do not pass a string containing `vicregl` (capitalization is ignored) to the `--backbone` argument.  
  *and / or*  
  - Alter the `src/models/backbone.py` file ([see here](https://github.com/julien-marteen-akay/Transferability-SSL-Wound-Recognition/blob/main/src/models/backbone.py)) and essentially remove the `get_vicregl_backbone` method of the `Backbone` class.

## Usage

1. **Create a Python environment**  
We used `Python 3.10.11`, but `3.9 <= version <= 3.11` should work, too.
2. Run `pip install -r requirements.txt`

### Data

We used two datasets in our work.
* AZH dataset
  1. Download from [here](https://github.com/uwm-bigdata/wound-classification-using-images-and-locations)
  2. Comes with a train/test split that is also reflected in our [.csv file](https://github.com/julien-marteen-akay/Transferability-SSL-Wound-Recognition/blob/main/data/Multi-modal-wound-classification-using-images-and-locations/dataframe.csv)  
  Note that the path column contains relative paths, i.e. replicate this structure from inside the `data/Multi-modal-wound-classification-using-images-and-locations` folder. The encoded labels' names can be looked up in the [json](https://github.com/julien-marteen-akay/Transferability-SSL-Wound-Recognition/blob/main/data/Multi-modal-wound-classification-using-images-and-locations/label_index_to_name.csv).
* Medetec Wound Database
  1. Download from [here](https://www.medetec.co.uk/files/medetec-image-databases.html)
  2. Create a similar .csv file (see the `Medetec` class in `src/datasets/wrappers.py`). Needs a column
  named `class` (actual class names; will be encoded internally) and a column named `path` (absolute paths to the images)

### Training

* For example to replicate the best ablation result on the AZH dataset, run the following command:  
```python
python -m src.training.training --custom_name my_replicated_experiment --dataset_name azh --backbone vicregl_convnext_xlarge --freeze_backbone true --num_mlp_layers 1 --gpu [0] --min_epochs 60 --epochs 300 --num_workers -1
```
* For more information on the arguments, run `python -m src.training.training -h`

### Evaluation

* In the `ablation_results` folder you will find our results.
* Running the training procedure, will also take care of testing the best model checkpoint.
* The results of the test and all the hyperparameters of the training process will be saved in an `experiment_info.csv` file.
* Results will be written into a folder named `out` created in the root directory of this project. Its overall directory
structure will be the same as in the folder `ablation_results`, i.e. `out/<dataset_name>/<backbone>/<custom_name>-<date-time stamp>`

## Citation

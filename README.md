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

- **CC BY-NC 4.0 Licensed Models**: Some models that can be downloaded and used in this repository are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). These models are for non-commercial use only. Attribution must be provided as specified in the license.

### Attribution for CC BY-NC 4.0 Licensed Models

All models from the repository accompanying the [VICRegL paper](https://arxiv.org/abs/2210.01571) are licensed under CC BY-NC 4.0.
- Original Authors: Adrien Bardes and Jean Ponce and Yann LeCun
- Source: https://github.com/facebookresearch/VICRegL
- License: https://github.com/facebookresearch/VICRegL/blob/main/LICENSE
- **<u>How to avoid using these models</u>:**
  - When starting the training process, do not pass a string containing `vicregl` (capitalization is ignored) to the `--backbone` argument.  
  *and / or*  
  - Alter the `src/models/backbone.py` file ([link](https://github.com/julien-marteen-akay/Transferability-SSL-Wound-Recognition/blob/main/src/models/backbone.py)) and essentially remove the `get_vicregl_backbone` method of the `Backbone` class.

## Usage

### Data

### Training

### Evaluation

## Citation

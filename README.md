# LandCover Segmentation


## **Overview**
> Simple PyTorch repository for density/segmentation tasks with PyTorch Lightning, Hydra and Neptune included.

## Table of contents
* [Requirements](#Requirements)
* [Structure](#Structure)
* [Usage](#Usage)

## Requirements

```bash
pip3 install -r requirements.txt
```


## Usage

* keys
  ```commandline
    export NEPTUNE_API_TOKEN=""
    export NEPTUNE_PROJECT_NAME=""
    ```
  
* prepare dataset
  ```commandline
  mkdir data/LandCover
  cd data/LandCover
  wget "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
  unzip landcover.ai.v1.zip
  python3 split.py
  ```
  
* run train
  ```commandline
  python run.py name=experiment_name
  ```
  
* run eval
  ```commandline
  python run.py name=experiment_name eval_mode=True trainer.resume_from_checkpoint=./path/to/model
  ```

* run export to ONNX
  ```commandline
  python run.py name=experiment_name eval_mode=True trainer.resume_from_checkpoint=./path/to/model export.export_to_onnx=True
  ```
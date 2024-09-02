# Road Damage Detection Using Ensemble YOLO
This repository contains the source code for our proposed model in [Optimized Road Damage Detection Challenge (ORDDC'2024)](https://orddc2024.sekilab.global/overview/). Our team addressed the cross-country road damage detection challenge by developing an ensemble model comprised of four YOLO models. We optimized the model by experimenting with various hyperparameters and training data, ultimately arriving at our most effective solution.

## Setup
Conda environment
```
conda create --name ensemble-yolo python=3.10.12 -y
conda activate ensemble-yolo
```
Clone this repository
```
git clone https://github.com/yinloonkhor/DoubleY_ORDCC2024.git
cd ensemble-yolo
```
Install dependencies
```
pip install -r requirements.txt 
```
Setup dataset
```
# Run prepare_data.ipynb
```
Train
```
python train.py 
```
Evaluation
```
python predict.py --model_path <model_directory> --source_path <source_image_directory> --output_path <output_directory>
```
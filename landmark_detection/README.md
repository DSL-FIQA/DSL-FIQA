# Landmark Detection Model
This page contains a pipeline for cropping, aligning, and detecting landmarks in custom datasets.

## Getting Started
Navigate to the `landmark_detection` directory:
```
cd landmark_detection
```
## Checkpoints
Download the necessary checkpoints with the following command:
```
python download_checkpoint.py
```

## Dataset
Please put your custom data in the folder `../dataset/custom/img`.

### Prediction
```
python landmark_detect.py --img_size 512 --bbox_margin_percentage 100
```
The landmark detection results will be saved in `../dataset/custom/landmark`.

## Acknowledgements
We thank the authors of [FaceXFormer](https://github.com/Kartik-3004/facexformer), from which this pipeline is based.

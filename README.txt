This repository is for CS230 project by Cheng Chen and Pablo Diaz. The objective of the model is to perform segmentaion of fetal brain in MRI scans for prenatal screening.

The unet model is in the file model_unet.py
The segnet model is in the file model_segnet.py

To run the training without early stopping, use python main.py
To run the training with early stopping, use python main_early_stopping.py
To run evaluation, use python eval.py

To change between unet and segnet model, using model = unet() or model = segnet() in the main script.

The model is in reference to https://github.com/zhixuhao/unet and https://github.com/ykamikawa/SegNet

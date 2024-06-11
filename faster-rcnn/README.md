## The Structure of this Repo(Custom files aren't attached)

```
├── README.md
├── VOCdevkit: to put the VOC dataset in.
├── frcnn.py
├── get_map.py
├── img: image for prediction.
│   ├── bus.jpg
│   ├── bus_predict.jpg
│   ├── cat.jpg
│   ├── cat_predict.jpg
│   ├── midway.jpg
│   └── midway_predict.jpg
├── logs: logs
│   └── best_epoch_weights.pth
├── model_data: store the model data, and dataset classes.
│   ├── resnet50-19c8e357.pth
│   ├── simhei.ttf
│   ├── voc_classes.txt
│   └── voc_weights_resnet.pth
├── nets
│   ├── __init__.py
│   ├── classifier.py
│   ├── frcnn.py
│   ├── frcnn_training.py
│   ├── resnet50.py
│   ├── rpn.py
│   └── vgg16.py
├── predict.py
├── proposal_box.py
├── summary.py
├── train.py: run this to train the model
├── utils
│   ├── __init__.py
│   ├── anchors.py
│   ├── callbacks.py
│   ├── dataloader.py
│   ├── util_miou.py
│   ├── utils.py
│   ├── utils_bbox.py
│   ├── utils_fit.py
│   └── utils_map.py
└── voc_annotation.py: run this script first.
```

## Training and Predicting Process

1. Since we cannot upload the VOC dataset into Github, so before training, you need to download the VOC dataset and unzip it to folder VOCdevkit.

2. Run voc_annotation.py to generate 2007_train.txt and 2007_val.txt.

3. Run train.py to start training.

4. To predict the new image, change the model_path to the model path, and classes_path to the classes path in file frcnn.py. After that, run predict.py and enter the image path to get the prediction. The image after prediction is saved in the same folder as the input image.

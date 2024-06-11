import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_pascal_voc

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.utils.visualizer import Visualizer
import cv2
import random

print("Detectron2 version:", detectron2.__version__)
def register_voc_datasets():
    for split in ['trainval', 'test']:
        dataset_name = "voc_2007_" + split
        if dataset_name not in DatasetCatalog.list():
            register_pascal_voc(dataset_name, "VOC2007", split, "./VOCdevkit")
        else:
            print("Dataset already registered")

register_voc_datasets()


print("Initializing training...")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("voc_2007_trainval",)
cfg.DATASETS.TEST = ("voc_2007_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # VOC has 20 classes
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
print("Training...")
trainer.train()

evaluator = COCOEvaluator("voc_2007_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "voc_2007_test")
inference_on_dataset(trainer.model, val_loader, evaluator)


dataset_dicts = DatasetCatalog.get("voc_2007_test")
# for d in random.sample(dataset_dicts, 4):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("voc_2007_test"), scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('Proposals', vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)


from BPlanDataset import BPlanDataset, train_test_split
from torch.utils.data import Dataset, DataLoader
from tools.utils import collate_fn
from tools.helper import get_transform
from mean_average_precision import MeanAveragePrecision2d
import torch
import numpy as np
from torchvision.ops import nms

def metric(model, dataset_test, num_class, device):
    
    metric_fn = MeanAveragePrecision2d(num_class)

    for i in range(len(dataset_test)):
        if i < 20:
            with torch.no_grad():
                pred = model([dataset_test[i][0].to(device)])
    keep = nms(pred[0]['boxes'], pred[0]['scores'], iou_threshold=0.5)
    keep = keep.tolist()
    

    gts = dataset_test[i][1]['boxes'].tolist()
    labels = [[el] for el in dataset_test[i][1]['labels'].tolist()]
    gts = np.array([gt + label + [0] + [0] for gt, label in zip(gts, labels)])

    preds_ = []
    bbox_after_nms = []
    labels_after_nms =[]
    scores_after_nms = []
    pred_list = pred[0]['boxes'].tolist()
    label_list = pred[0]['labels'].tolist()
    scores_list = pred[0]['scores'].tolist()
    for idx in keep:
        bbox_after_nms.append(pred_list[idx])
        scores_after_nms.append(scores_list[idx])
        labels_after_nms.append(label_list[idx])
    #preds_ = pred[0]['boxes'].tolist()
    
    
    
    pred_labels = [[el] for el in labels_after_nms]
    scores = [[el] for el in scores_after_nms]
    preds = np.array([pred + label + score for pred, label, score in zip(bbox_after_nms, pred_labels, scores)])
    print(preds)
    print(gts)
    metric_fn.add(preds, gts)


    
    # compute PASCAL VOC metric
    print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

    # compute PASCAL VOC metric at the all points
    print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

    # compute metric COCO metric
    print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_object_detection(num_classes, dim_reduced=256):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=500)

    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=num_classes)

    return model

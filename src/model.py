import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights


def init_model(args):
    num_classes = 12  # 11 classes + background
    model_type = args.model_type
    if model_type == 'resnet50':
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # freeze the entire model
        for parameter in model.parameters():
            parameter.requires_grad = False

        # unfreeze the predictor
        for parameter in model.roi_heads.box_predictor.parameters():
            parameter.requires_grad = True

        transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
    else:
        raise NotImplementedError(f"model type not recognized {model_type}")

    return model, transforms
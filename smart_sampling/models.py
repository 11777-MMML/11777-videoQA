import torch
from torch import nn
from torchvision import transforms as T
from transformers import AutoFeatureExtractor, ViTMAEModel
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor

# Taken from: https://pytorch.org/hub/pytorch_vision_resnet/
HUB_URL = 'pytorch/vision:v0.10.0'
RESNET_LAYERS = [34, 50, 101, 152]

# Taken from: https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEModel
MAE_CHECKPOINT = 'facebook/vit-mae-base'

class NextFramePredictor(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class FrameEmbedder(nn.Module):
    def __init__(self, model_name: str, **kwargs):
        self.model = None
        self.feature_extractor = None

        if model_name.startswith("resnet"):
            self.resnet(num_layers=kwargs['num_layers'])
        elif model_name.startswith("mae"):
            self.mae()
        else:
            raise ValueError(f"{model_name} is not supported")
    
    def resnet(self, num_layers):
        if num_layers not in RESNET_LAYERS:
            raise ValueError(f"{num_layers} not in {RESNET_LAYERS}")

        model_string = f"resnet{num_layers}"
        self.model = torch.hub.load(HUB_URL, model_string, pretrained=True)

        # Taken from: https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
        # Drop the last fc layer in layer4
        self.model.layer4 = nn.Sequential(*list(self.model.layer4.children()[:-1]))

        # Taken from: https://pytorch.org/hub/pytorch_vision_resnet/
        # TODO: Do we need centercrop?
        self.feature_extractor = T.Compose(
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    def mae(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MAE_CHECKPOINT)
        self.model = ViTMAEModel.from_pretrained(MAE_CHECKPOINT)
    
    def forward(self, images):
        if isinstance(self.feature_extractor, ViTFeatureExtractor):
            # Taken from: https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEModel.forward.example
            inputs = self.feature_extractor(images=images, return_tensors='pt')
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        else:
            inputs = self.feature_extractor(images)
            outputs = self.model(inputs)
            last_hidden_state = outputs
        
        return last_hidden_state
        
class Agent(nn.Module):
    def __init__(self):
        pass

    def forward():
        pass

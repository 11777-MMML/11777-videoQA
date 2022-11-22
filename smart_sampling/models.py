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

class SequenceModel(nn.Module):
    def __init__(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

class TransformerModel(nn.Module):
    def __init__(self, max_seq_len=64, input_dim=768, num_heads=3, num_layers=3, activation='gelu'):
        super(TransformerModel, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation=activation,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

        self.max_seq_len = max_seq_len
    
    def forward(self, input, memory, seq_len):
        mask = torch.zeros_like(input)
        mask[:, 0:seq_len] = 1

        out =  self.decoder(
            tgt=input,
            memory=memory,
            tgt_make=mask,
            memory_mask=mask,
        )

        return out

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
    def __init__(self, num_layers, input_dim, hidden_dim, activation):
        # Policy Network
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = 2
        layers = []

        # Define all but one
        for _ in range(num_layers - 1):
            layer = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
            input_dim = hidden_dim
            layers.append(layer, activation)
        
        # This model outputs logits
        layer = nn.Linear(in_features=input_dim, out_features=self.out_dim)
        layers.append(layer)

        self.policy_network = nn.Sequential(*layers)
        
    def forward(self, input_obs):
        return self.policy_network(input_obs)

import torch
from typing import Union
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import AutoFeatureExtractor, ViTMAEModel
from transformers.models.vit.feature_extraction_vit import ViTFeatureExtractor

# Taken from: https://pytorch.org/hub/pytorch_vision_resnet/
HUB_URL = 'pytorch/vision:v0.10.0'
RESNET_LAYERS = [34, 50, 101, 152]

# Taken from: https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEModel
MAE_CHECKPOINT = 'facebook/vit-mae-base'

class StateConfig:
    def __init__(self):
        self.d_model = 768
        self.n_head = 8
        self.activation = 'gelu'
        self.n_layers = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionConfig:
    def __init__(self):
        self.d_model = 768
        self.n_head = 8
        self.activation = 'gelu'
        self.n_layers = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentConfig:
    def __init__(self):
        self.num_layers=2 
        self.input_dim=768 
        self.hidden_dim=512 
        self.activation=torch.nn.LeakyReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceModel(nn.Module):
    def __init__(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

class TransformerModel(nn.Module):
    def __init__(self, config: Union[StateConfig, PredictionConfig]):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            activation=config.activation
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.n_layers,
        )

        self.device = config.device
    
    def forward(self, input):
        input = input.to(self.device)
        
        out = self.transformer(input)

        return out

class StateModel(nn.Module):
    def __init__(self, config: StateConfig) -> None:
        super().__init__()

        self.transformer = TransformerModel(config=config)
        self.device = config.device
    
    def forward(self, input):
        input = torch.concat(input)
        out = self.transformer(input)

        return out
    
class PredictionModel(nn.Module):
    def __init__(self, config: PredictionConfig) -> None:
        super().__init__()

        self.transformer = TransformerModel(config=config)
        self.device = config.device
    
    def forward(self, frames, question, candidates):
        num_frames = len(frames)
        len_question = len(question)
        len_candidates = len(candidates)

        rep = torch.cat(frames + question + candidates)

        rep = rep.to(self.device)
        out = self.transformer(rep)
        out_q = out[num_frames + 1]
        out_candidates = out[-len_candidates:]

        return F.cosine_similarity(out_q, out_candidates)

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
    def __init__(self, config: AgentConfig):
        super().__init__()
        # Policy Network
        self.num_layers = config.num_layers
        input_dim =  config.input_dim
        self.input_dim = input_dim
        self.hidden_dim = config.hidden_dim
        self.device = config.device
        self.out_dim = 2
        layers = []

        # Define all but one
        for _ in range(config.num_layers - 1):
            layer = nn.Linear(in_features=input_dim, out_features=config.hidden_dim, bias=True)
            input_dim = config.hidden_dim
            layers.append(layer)
            layers.append(config.activation)
        
        # This model outputs logits
        layer = nn.Linear(in_features=input_dim, out_features=self.out_dim)
        layers.append(layer)

        self.policy_network = nn.Sequential(*layers)
        self.policy_network = self.policy_network.to(self.device)
        
    def forward(self, input_obs):
        return self.policy_network(input_obs)

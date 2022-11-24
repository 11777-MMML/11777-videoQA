import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math
from collections import OrderedDict
from enum import IntEnum

@dataclass
class Config:
    # ATPEncoder params
    n_layers: int = 8
    frame_input_dim = 4096
    n_frames = 16
    qa_input_dim = 768
    n_heads: int = 8
    d_model: int = 512
    d_model_ff: int = 512
    enc_dropout: float = 0.1
    n_cands: int = 5
    
    @classmethod
    def from_args(cls, args):
        return cls(n_layers = args.n_layers,
                   n_heads = args.n_heads,
                   d_model = args.d_model,
                   d_model_ff = args.d_model_ff,
                   enc_dropout = args.enc_dropout,
                   n_cands = args.n_cands)

class ModalityEmbeddingsID(IntEnum):
    TEXT_EMBEDDING = 0
    VISUAL_EMBEDDING = 1

class ModalityEmbeddings(nn.Module):
    def __init__(self,
                 d_model: int):
        """
        Details for each of these arguments are provided in ATPConfig.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=len(ModalityEmbeddingsID),
                                      embedding_dim=d_model)

    def forward(self, class_ids: torch.tensor):
        # return modality embeddings
        return self.embedding(class_ids)


class FrameQAReasonLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.qa_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        self.frame_encoder = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        self.frame_qa_condition = nn.MultiheadAttention(self.d_model, config.n_heads)
        self.qa_frame_condition = nn.MultiheadAttention(self.d_model, config.n_heads)

    def forward(self, x):
        frame_inputs, qa_inputs = x
        qa_inputs = self.qa_encoder_layer(qa_inputs)
        # q_input = qa_inputs[0].unsqueeze(0)
        # a_inputs = qa_inputs[1:]
        frame_qa_condition, _ = self.frame_qa_condition(frame_inputs, qa_inputs, qa_inputs)
        frame_inputs = frame_inputs + frame_qa_condition
        frame_inputs = self.frame_encoder(frame_inputs)
        qa_frame_condition, _ = self.qa_frame_condition(qa_inputs, frame_inputs, frame_inputs)
        qa_inputs = qa_inputs + qa_frame_condition
        # qa_inputs = torch.cat((q_input, a_inputs), dim=0)
        return (frame_inputs, qa_inputs)

class TemporalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FrameQAReasonModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.frame_projection = nn.Linear(config.frame_input_dim, config.d_model)
        self.qa_projection = nn.Linear(config.qa_input_dim, config.d_model)
        self.video_temporal_encoder = TemporalEncoding(config.frame_input_dim, max_len=config.n_frames)
        self.logits = nn.MultiheadAttention(config.d_model, config.n_heads) 
        # self.logits = nn.Linear(config.d_model, 1)

        self.reasoning_module = nn.Sequential(OrderedDict(
                [(f"FrameQAReasonLayer_{x}", FrameQAReasonLayer(self.config)) for x in range(config.n_layers)]
            )
        )
        self.modality_embedding_layer = ModalityEmbeddings(config.d_model)

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_qa: torch.tensor,
                **kwargs):      
        # N, L, D = x_vis_seq.size()  # (batch_size, sequence_length, feature_dimension)
        x_vis_seq = x_vis_seq.permute(1, 0, 2)    # make (L, N, D); sequence first

        x_vis_seq = self.video_temporal_encoder(x_vis_seq)
        x_vis_seq = self.frame_projection(x_vis_seq) 
        
        # visual_class_ids = torch.tensor([ModalityEmbeddingsID.VISUAL_EMBEDDING] * L, dtype=torch.long, device=x_vis_seq.device).unsqueeze(-1)
        # x_vis_seq = x_vis_seq + self.modality_embedding_layer(visual_class_ids)

        # N, L, D = x_txt_qa.size()  

        x_txt_qa = x_txt_qa.permute(1, 0, 2) 
        x_txt_qa = self.qa_projection(x_txt_qa) 

        # text_class_ids = torch.tensor([ModalityEmbeddingsID.TEXT_EMBEDDING] * L, dtype=torch.long, device=x_txt_qa.device).unsqueeze(-1)
        # x_txt_qa = x_txt_qa + self.modality_embedding_layer(text_class_ids)

        # x_vis_seq = torch.cat((x_vis_seq, x_txt_qa[0].unsqueeze(0)), dim=0)
        # x_txt_qa = x_txt_qa[1:]

        x_vis_seq, x_txt_qa = self.reasoning_module((x_vis_seq, x_txt_qa))


        # visual_rep, scores = self.logits(x_txt_qa[0].unsqueeze(0), x_vis_seq, x_vis_seq)
        text_rep, scores = self.logits(x_vis_seq, x_txt_qa, x_txt_qa)
        # logits = F.cosine_similarity(visual_rep, x_txt_qa[1: ],dim=-1)
        # logits = self.logits(x_txt_qa)
        # logits = logits.squeeze().transpose(0,1)

        logits = scores.mean(dim=1)

        
        return logits
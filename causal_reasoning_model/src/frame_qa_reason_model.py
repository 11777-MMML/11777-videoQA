import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class Config:
    # ATPEncoder params
    n_layers: int = 3
    frame_input_dim = 768
    n_frames = 16
    qa_input_dim = 512
    n_heads: int = 2
    d_model: int = 256
    d_model_ff: int = 256
    enc_dropout: float = 0.1
    n_cands: int = 5
    
    @classmethod
    def from_args(cls, args):
        return cls(n_layers = args.n_layers,
                   n_heads = args.n_heads,
                   d_model = args.d_model,
                   d_model_ff = args.d_model_ff,
                   enc_dropout = args.enc_dropout,
                   use_text_query = args.use_text_query,
                   use_text_cands = args.use_text_cands,
                   n_cands = args.n_cands)

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

    def forward(self, frame_inputs, qa_inputs):
        qa_inputs = self.qa_encoder_layer(qa_inputs)
        frame_qa_condition = self.frame_qa_condition(frame_inputs, qa_inputs)
        frame_inputs += frame_qa_condition
        frame_inputs = self.frame_encoder(frame_inputs)
        qa_frame_condition = self.qa_frame_condition(qa_inputs, frame_inputs)
        qa_inputs += qa_frame_condition
        return frame_inputs, qa_inputs

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
        self.dropout = nn.Dropout(p=config.sel_dropout)
        self.logits = nn.Linear(config.d_model, 1)
        self.reasoning_module = nn.Sequential(
            [FrameQAReasonLayer(self.config) for x in range(config.n_layers)]
        )

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_qa: torch.tensor,
                **kwargs):      
        N, L, D = x_vis_seq.size()  # (batch_size, sequence_length, feature_dimension)
        x_vis_seq = x_vis_seq.permute(1, 0, 2)    # make (L, N, D); sequence first

        x_vis_seq = self.frame_projection(x_vis_seq) * math.sqrt(self.d_model)
        x_vis_seq = self.video_temporal_encoder(x_vis_seq)

        x_txt_qa = x_txt_qa.permute(1, 0, 2) 
        x_txt_qa = self.qa_projection(x_txt_qa)

        x_vis_seq, x_txt_qa = self.reasoning_module(x_vis_seq, x_txt_qa)

        logits = self.logits(x_txt_qa)
        return logits
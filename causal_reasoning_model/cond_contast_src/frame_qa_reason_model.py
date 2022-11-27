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
    frame_input_dim = 768
    n_frames = 16
    qa_input_dim = 512
    n_heads: int = 8
    d_model: int = 512
    d_model_ff: int = 512
    enc_dropout: float = 0.1
    n_answers: int = 5
    n_cands: int = 0
    
    @classmethod
    def from_args(cls, args):
        return cls(n_layers = args.n_layers,
                   n_heads = args.n_heads,
                #    d_model = args.d_model,
                #    d_model_ff = args.d_model_ff,
                   enc_dropout = args.enc_dropout,
                   n_answers = args.n_answers,
                   n_cands = args.n_cands)

class ModalityEmbeddingsID(IntEnum):
    TEXT_EMBEDDING_Q = 0
    TEXT_EMBEDDING_A = 1
    TEXT_EMBEDDING_C = 2
    VISUAL_EMBEDDING = 3

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
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.d_model,
        #     nhead=config.n_heads,
        #     dim_feedforward=config.d_model_ff,
        #     dropout=config.enc_dropout,
        #     activation='gelu'
        # )

        self.question_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        self.vision_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        self.answer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model_ff,
            dropout=config.enc_dropout,
            activation='relu'
        )
        # self.vision_text_attention = nn.MultiheadAttention(self.d_model, config.n_heads)
        # self.text_vision_attention = nn.MultiheadAttention(self.d_model, config.n_heads)

        self.vision_pre_norm = nn.LayerNorm(self.d_model)
        self.question_pre_norm = nn.LayerNorm(self.d_model)
        self.answer_pre_norm = nn.LayerNorm(self.d_model)

        self.vision_attention = nn.MultiheadAttention(self.d_model, config.n_heads)
        self.question_attention = nn.MultiheadAttention(self.d_model, config.n_heads)
        self.answer_attention = nn.MultiheadAttention(self.d_model, config.n_heads)

    def forward(self, x):
        vision_inputs, question_inputs, answer_inputs = x

        vision_inputs = self.vision_pre_norm(vision_inputs)
        question_inputs = self.question_pre_norm(question_inputs)
        answer_inputs = self.answer_pre_norm(answer_inputs)
        
        # q_input = qa_inputs[0].unsqueeze(0)
        # a_inputs = qa_inputs[1:]
        qa_inputs = torch.cat((question_inputs, answer_inputs))
        vq_inputs = torch.cat((vision_inputs, question_inputs))
        av_inputs = torch.cat((answer_inputs, vision_inputs))

        v_condition, _ = self.vision_attention(vision_inputs, qa_inputs, qa_inputs)
        q_condition, _ = self.question_attention(question_inputs, av_inputs, av_inputs)
        a_condition, _ = self.answer_attention(answer_inputs, vq_inputs, vq_inputs)

        # vision_text_condition, _ = self.vision_text_attention(vision_inputs, text_inputs, text_inputs)
        # vision_inputs = vision_inputs + vision_text_condition
        # vision_inputs = self.vision_encoder_layer(vision_inputs)
        # text_vision_condition, _ = self.text_vision_attention(text_inputs, vision_inputs, vision_inputs)
        # text_inputs = text_inputs + text_vision_condition
        # text_inputs = self.text_encoder_layer(text_inputs)
        # qa_inputs = torch.cat((q_input, a_inputs), dim=0)
        vision_inputs  = self.vision_encoder_layer(vision_inputs + v_condition)
        question_inputs = self.question_encoder_layer(question_inputs + q_condition)
        answer_inputs = self.answer_encoder_layer(answer_inputs + a_condition)

        # vision_inputs 
        return (vision_inputs, question_inputs, answer_inputs)

    # def forward(self, x):
    #     x = self.encoder_layer(x)
    #     return x

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
        # self.a_projection = nn.Linear(config.qa_input_dim, config.d_model)
        self.video_temporal_encoder = TemporalEncoding(config.d_model, max_len=config.n_frames)
        self.vis_q_condition = nn.MultiheadAttention(config.d_model, config.n_heads) 
        self.logits = nn.Linear(config.d_model, 1)

        self.reasoning_module = nn.Sequential(OrderedDict(
                [(f"FrameQAReasonLayer_{x}", FrameQAReasonLayer(self.config)) for x in range(config.n_layers)]
            )
        )
        self.modality_embedding_layer = ModalityEmbeddings(config.d_model)

    def forward(self,
                x_vis_seq: torch.tensor,
                x_txt_qa: torch.tensor,
                **kwargs):      
        N, L, D = x_vis_seq.size()  # (batch_size, sequence_length, feature_dimension)
        x_vis_seq = x_vis_seq.permute(1, 0, 2)    # make (L, N, D); sequence first


        x_vis_seq = self.frame_projection(x_vis_seq) 
        x_vis_seq = self.video_temporal_encoder(x_vis_seq)
        vision_cls_ids = [ModalityEmbeddingsID.VISUAL_EMBEDDING] * L
        vision_cls_ids = torch.tensor(vision_cls_ids, dtype=torch.long, device=x_vis_seq.device).unsqueeze(-1)
        x_vis_seq = x_vis_seq + self.modality_embedding_layer(vision_cls_ids)

   

        x_txt_qa = x_txt_qa.permute(1, 0, 2) 
        x_txt_qa = self.qa_projection(x_txt_qa) # quesiton, 5 answers, 25 candidates 

        x_question = x_txt_qa[0].unsqueeze(0)
        question_cls_ids = [ModalityEmbeddingsID.TEXT_EMBEDDING_Q]
        question_cls_ids = torch.tensor(question_cls_ids, dtype=torch.long, device=x_question.device).unsqueeze(-1)
        x_question = x_question + self.modality_embedding_layer(question_cls_ids)

        x_ans = x_txt_qa[1:]
        answer_cls_ids = [ModalityEmbeddingsID.TEXT_EMBEDDING_A]*self.config.n_answers
        answer_cls_ids = torch.tensor(answer_cls_ids, dtype=torch.long, device=x_ans.device).unsqueeze(-1)
        x_ans = x_ans + self.modality_embedding_layer(answer_cls_ids)

        # x_vis_ans = torch.cat((x_vis_seq, x_ans), dim = 0)


        # class_ids = [ModalityEmbeddingsID.TEXT_EMBEDDING_Q]
        # class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING_A]*self.config.n_answers)
        # # class_ids.extend([ModalityEmbeddingsID.TEXT_EMBEDDING_C]*self.config.n_cands)
        # class_ids = torch.tensor(class_ids, dtype=torch.long, device=x_vis_seq.device).unsqueeze(-1)
        # x_txt_qa = x_txt_qa + self.modality_embedding_layer(class_ids)

        # x_vis_ans = x_vis_ans + self.modality_embedding_layer(class_ids)

        x_input = (x_vis_seq, x_question, x_ans) # torch.cat((x_vis_seq, x_txt_qa), dim=0)

        # x_input = (x_question, x_vis_ans)

        x_vis, x_question, x_ans = self.reasoning_module(x_input)

        # x_question, x_vis_ans = self.reasoning_module(x_input)

        # x_vis = x_vis_ans[:len(x_vis_ans) - (self.config.n_answers)]
        # x_question = x_txt_qac[0].unsqueeze(0)
        # x_ans = x_vis_ans[len(x_vis):(self.config.n_answers + len(x_vis))]
        # x_question = x_txt_qa[0].unsqueeze(0)
        # x_ans = x_txt_qa[1:]
        x_cands = [] # x_txt_qac[-(self.config.n_cands):]

        # assert len(x_vis) + len(x_question) + len(x_ans) + len(x_cands) == len(x_input)

        # x_vis_q_condition, _ = self.vis_q_condition(x_question, x_vis, x_vis)





        # visual_rep, scores = self.logits(x_txt_qa[0].unsqueeze(0), x_vis_seq, x_vis_seq)
        # text_rep, scores = self.logits(x_vis_seq, x_txt_qa, x_txt_qa)
        logits = F.cosine_similarity(x_question, x_ans, dim=-1)
        # logits = self.logits(x_ans)
        # print(logits.shape)
        # logits = logits.squeeze()#.transpose(0,1)

        # logits = scores.mean(dim=1)

        
        return logits, x_question, x_ans, x_cands
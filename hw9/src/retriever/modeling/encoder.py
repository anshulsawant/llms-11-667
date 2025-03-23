from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoModel

from transformers.file_utils import ModelOutput
from retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder: PreTrainedModel,
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.temperature = temperature

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_text(query) if query else None
        p_reps = self.encode_text(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        scores = self.compute_similarity(q_reps, p_reps, self.temperature)
        target = self.compute_labels(q_reps.size(0), p_reps.size(0)).to(scores.device)
        loss = self.compute_loss(scores, target)

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    
    def pooling(self, last_hidden_state, attention_mask):

        """
        Use the atention mask to find the index of each sequence's last token;
        Perform last-token pooling.
        Apply L2 norm to the embeddings.  
        
        Args:
        - last_hidden_state:  tensor of shape (batch_size, seq_len, hidden_dim)
        - attention_mask:  tensor of shape (batch_size, seq_len)

        Returns:
        - reps: tensor of shape (batch_size, hidden_dim)
        """
        raise NotImplementedError()

    def encode_text(self, text):
        hidden_states = self.encoder(**text, return_dict=True)
        hidden_states = hidden_states.last_hidden_state
        return self.pooling(hidden_states, text['attention_mask'])

    def compute_similarity(self, q_reps, p_reps, temperature):
        """
        Compute the dot product between q_reps and p_reps.
        Apply temperature.

        Note that n_queries == batch_size and n_passages == batch_size * train_group_size,
        where train_group_size is the parameter stating how many passages per query are used.  

        Args:
        - q_reps:  tensor of shape (n_queries, hidden_dim)
        - p_reps:  tensor of shape (n_passages, seq_len)
        - temperature:  float

        Returns:
        - similarity_matrix: tensor of shape (n_queries, n_passages)
        """
        raise NotImplementedError()

    def compute_labels(self, n_queries, n_passages):
        """
        Compute the labels array.

        n_passages is the total number of passages.  
        Hence, the number of passages per query is n_passages // n_queries.
        Out of each group of n_passages // n_queries, the first is the positive for the respective query.

        For example, 2 queries with 2 passages per query:

        similarity_matrix = [[1, 2, 3, 4],
                             [5, 6, 7, 8]]

        n_queries = 2
        n_passages = 4
        expected_labels = [0, 3]:
            - for query 1, the 0th entry is the positive ([1, 2, 3, 4] : 1)
            - for query 2, the 3rd entry is the positive ([5, 6, 7, 8] : 7)
    

        Args:
        - n_queries:  int
        - n_passages:  int

        Returns:
        - target: tensor of shape (n_queries)
        """
        raise NotImplementedError()
    
    def compute_loss(self, scores, target):
        """
        Compute the mean reduced loss.
        
        Args:
        - scores:  tensor of shape (n_queries, n_passages)
        - target:  tensor of shape (n_queries)

        Returns:
        - loss: mean reduced loss.
        """
        raise NotImplementedError()

    def gradient_checkpointing_enable(self, **kwargs):
        try:
            self.encoder.model.gradient_checkpointing_enable()
        except Exception:
            self.encoder.gradient_checkpointing_enable()


    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        model = cls(
            encoder=base_model,
            temperature=model_args.temperature
        )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        model = cls(
            encoder=base_model,
        )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
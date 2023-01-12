# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : yingjie.xu
# @File : model.py
# @Description: torch 实现 UIE模型架构
# @Reference: https://github.com/PaddlePaddle/PaddleNLP/tree/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie


import torch.nn as nn
import torch


class UIE(nn.Module):
	__doc__ = """The structure of UIE model..."""

	def __init__(self, encoder, hidden_size=768):
		"""
		:param encoder: 以 transformers.AutoModel 为 backbone, 默认使用 ernie3.0
		:param hidden_size: encoder 最后一层的维度
		"""
		super().__init__()
		self.encoder = encoder
		self.linear_start = nn.Linear(hidden_size, 1)
		self.linear_end = nn.Linear(hidden_size, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self,
	            input_ids: torch.tensor,
	            token_type_ids: torch.tensor,
	            attention_mask=None,
	            pos_ids=None) -> tuple:
		sequence_output = self.encoder(
			input_ids=input_ids,
			token_type_ids=token_type_ids,
			position_ids=pos_ids,
			attention_mask=attention_mask
		)["last_hidden_state"]

		start_logits = self.linear_start(sequence_output)  # (batch, seq_len, 1)
		start_logits = torch.squeeze(start_logits, -1)     # (batch, seq_len)
		start_prob = self.sigmoid(start_logits)            # (batch, seq_len)

		end_logits = self.linear_end(sequence_output)      # (batch, seq_len, 1)
		end_logits = torch.squeeze(end_logits, -1)         # (batch, seq_len)
		end_prob = self.sigmoid(end_logits)                # (batch, seq_len)

		return start_prob, end_prob

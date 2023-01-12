# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : yingjie.xu
# @File : metrics.py
# @Description: 常见指标评估
# @Reference: https://github.com/taishan1994/pytorch_uie_ner/blob/6af3d84a7037acd892b484ea630daa6b4a9b5815/utils.py#L146


import numpy as np

from processor import get_bool_ids_greater_than, get_span


class SpanEvaluator(object):
	__doc__ = """此类用于计算span的precision, recall and F1-score"""

	def __init__(self):
		super().__init__()
		self.num_infer_spans = 0
		self.num_label_spans = 0
		self.num_correct_spans = 0

	def compute(self, start_probs, end_probs, gold_start_ids, gold_end_ids):
		pred_start_ids = get_bool_ids_greater_than(start_probs)
		pred_end_ids = get_bool_ids_greater_than(end_probs)
		gold_start_ids = get_bool_ids_greater_than(gold_start_ids.tolist())
		gold_end_ids = get_bool_ids_greater_than(gold_end_ids.tolist())
		num_infer_spans = 0
		num_label_spans = 0
		num_correct_spans = 0

		for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
				pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids):
			_correct, _infer, _label = self.eval_span(predict_start_ids, predict_end_ids,
			                                          label_start_ids, label_end_ids)
			num_correct_spans += _correct
			num_infer_spans += _infer
			num_label_spans += _label

		return num_correct_spans, num_infer_spans, num_label_spans

	def eval_span(self, predict_start_ids, predict_end_ids, label_start_ids, label_end_ids):
		"""
		评估位置抽取(start, end)
		:param predict_start_ids: [1, 2, 10]
		:param predict_end_ids: [4, 12]
		:param label_start_ids: [2, 10]
		:param label_end_ids: [4, 11]
		:return: num_correct, num_infer, num_label，如(1, 2, 2)
		"""
		pred_set = get_span(predict_start_ids, predict_end_ids)  # 得到模型输出的span集合(set),如 {(1, 3), (4, 5)}
		label_set = get_span(label_start_ids, label_end_ids)  # 得到标签中正确的span集合(set),如 {(1, 3), (4, 5), (8, 9)}
		num_correct = len(pred_set & label_set)  # 计算正确预测的span集合(两个集合求交集),如 {(1, 3), {4, 5}}
		num_infer = len(pred_set)
		num_label = len(label_set)
		return num_correct, num_infer, num_label

	def accumulate(self):
		"""
		此函数返回所有累积小批次的平均 precision, recall and f1 score
		:return: tuple(`precision, recall, f1 score`)
		"""
		precision = float(self.num_correct_spans / self.num_infer_spans) if self.num_infer_spans else 0.
		recall = float(self.num_correct_spans / self.num_label_spans) if self.num_label_spans else 0.
		f1_score = float(2 * precision * recall / (precision + recall)) if self.num_correct_spans else 0.
		return precision, recall, f1_score

	def update(self, num_correct_spans, num_infer_spans, num_label_spans):
		"""
		This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
		to accumulate and update the corresponding status of the SpanEvaluator object.
		"""
		self.num_infer_spans += num_infer_spans
		self.num_label_spans += num_label_spans
		self.num_correct_spans += num_correct_spans

	def reset(self):
		"""
		Reset function empties the evaluation memory for previous mini-batches.
		"""
		self.num_infer_spans = 0
		self.num_label_spans = 0
		self.num_correct_spans = 0

	def name(self):
		"""
		Return name of metric instance.
		"""
		return "precision", "recall", "f1"


if __name__ == '__main__':
	start_ids = [1, 2, 10]
	end_ids = [4, 12]
	result = get_span(start_ids, end_ids)
	print(result)

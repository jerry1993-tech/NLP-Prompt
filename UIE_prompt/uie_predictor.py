# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : yingjie.xu
# @File : uie_predictor.py
# @Description: 利用训练的最佳UIE模型根据下游任务进行推理
import os
from typing import List

import torch
from transformers import AutoTokenizer

from processor import convert_inputs, get_bool_ids_greater_than, get_span

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")  # 指定运行设备
saved_model_path = './checkpoint/model_best'                          # 训练的最佳UIE模型地址
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = torch.load(os.path.join(saved_model_path, 'model.pt'))
model.to(device).eval()


def inference(contents: List[str], prompts: List[str], max_length=512, prob_threshold=0.5) -> list[list[str]]:
	"""
	对输入的 prompts 和 contents 进行预测，返回模型提取结果
	:param contents: (List[str])，待提取文本列表，如 [
                                                    '1月12日北京飞上海话费一百元',
                                                    '6月7日苏州街地铁站到海淀黄庄',
                                                    ...
                                                ]
	:param prompts: (List[str])，prompt列表，用于告知模型需提取的内容，如 [
                                                                    '时间',
                                                                    '出发地',
                                                                    ...
                                                                ]
	:param max_length: (int)，句子最大长度，小于最大长度则padding，大于最大长度则截断
	:param prob_threshold: (float)，sigmoid概率阈值，大于该阈值则二值化为True
	:return: 模型识别结果 -> [['昨天'], ['苏州街地铁站']]
	"""
	inputs = convert_inputs(tokenizer, prompts, contents, max_length)
	model_inputs = {
		'input_ids': inputs['input_ids'].to(device),
		'token_type_ids': inputs['token_type_ids'].to(device),
		'attention_mask': inputs['attention_mask'].to(device),
	}
	output_sp, output_ep = model(**model_inputs)
	output_sp, output_ep = output_sp.detach().cpu().tolist(), output_ep.detach().cpu().tolist()
	start_ids_list = get_bool_ids_greater_than(output_sp, prob_threshold)
	end_ids_list = get_bool_ids_greater_than(output_ep, prob_threshold)

	results = []                                     # decode模型输出，将token id转换为span text
	offset_mapping = inputs['offset_mapping'].tolist()
	for start_ids, end_ids, prompt, content, offset_map in zip(start_ids_list, end_ids_list, prompts, contents, offset_mapping):
		span_set = get_span(start_ids, end_ids)
		current_span_list = []
		for span in span_set:
			if span[0] < len(prompt) + 2:            # 若结果在 prompt 区则过滤掉
				continue
			span_text = ''                           # 结果 span
			input_content = prompt + content         # 对齐 token_ids
			for s in range(span[0], span[1] + 1):    # 将 offset map 里 token 转成对应的文本
				span_text += input_content[offset_map[s][0]: offset_map[s][1]]
			current_span_list.append(span_text)
		results.append(current_span_list)

	return results


def ner_example(sentence: str, schema: list, max_length=128, prob_threshold=0.6) -> dict:
	"""
	UIE 在 NER任务上的应用
	:param sentence: (str)待抽取的句子，如 '5月17号晚上10点35分加班从公司打车回家花费36块五。'
	:param schema: (list)待抽取的实体列表,如 ['时间', '出发地', '目的地', '费用']
	:param max_length: (int)，句子最大长度，小于最大长度则padding，大于最大长度则截断
	:param prob_threshold: (float, optional)置信度阈值（0~1），置信度越高则P越大R越小
	:return: dict -> {
                实体1: [实体值1, 实体值2, 实体值3...],
                实体2: [实体值1, 实体值2, 实体值3...],
                ...
            }
	"""
	entities = {}
	sentences = [sentence] * len(schema)    # 这里是UIE的核心：用不同的prompt输入代替了Multi Head，则一个prompt需要对应一个句子，所以要复制n遍句子
	result = inference(
		sentences,
		schema,
		max_length=max_length,
		prob_threshold=prob_threshold
	)
	for s, r in zip(schema, result):
		entities[s] = r

	return entities


def event_extract_example(sentence: str, schema: dict, max_length=128, prob_threshold=0.6) -> dict:
	"""
	UIE 在事件关系抽取任务上的应用
	:param sentence: (str)待抽取的句子，如 '5月17号晚上10点35分加班从公司打车回家花费36块五。'
	:param schema: (dict)事件定义字典，如 {
                                            '加班触发词': ['时间','地点'],
                                            '出行触发词': ['时间', '出发地', '目的地', '花费']
                                        }
	:param max_length: (int)，句子最大长度，小于最大长度则padding，大于最大长度则截断
	:param prob_threshold: (float, optional)置信度阈值（0~1），置信度越高则P越大R越小
	:return: dict -> {
                '触发词1': {},
                '触发词2': {
                    '事件属性1': [属性值1, 属性值2, ...],
                    '事件属性2': [属性值1, 属性值2, ...],
                    '事件属性3': [属性值1, 属性值2, ...],
                    ...
                }
            }
	"""
	rsp = {}
	trigger_prompts = list(schema.keys())
	for trigger_prompt in trigger_prompts:
		rsp[trigger_prompt] = {}
		triggers = trigger_prompt.rstrip('触发词')

		for trigger in triggers:
			if trigger:
				arguments = schema.get(trigger_prompt)
				contents = [sentence] * len(arguments)
				prompts = [f"{trigger}的{a}" for a in arguments]
				res = inference(
					contents,
					prompts,
					max_length=max_length,
					prob_threshold=prob_threshold
				)
				for a, r in zip(arguments, res):
					rsp[trigger_prompt][a] = r

	return rsp


if __name__ == '__main__':
	from rich import print
	contents = ['1月12日北京飞上海话费一百元', '6月7日苏州街地铁站到海淀黄庄']
	prompts = ['时间', '出发地']
	results = inference(contents, prompts, 50)
	print(results)

	# test NER
	sentence = '5月17号晚上10点35分加班从公司打车回家花费36块五。'
	entities = ner_example(
		sentence=sentence,
		schema=['时间', '出发地', '目的地', '费用']
	)
	print(entities)

	# test event extract
	rsp = event_extract_example(
		sentence=sentence,
		schema={
			'加班触发词': ['时间', '地点'],
			'出行触发词': ['时间', '出发地', '目的地', '花费']
		}
	)
	print(rsp)

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : yingjie.xu
# @File : utils.py
# @Description: 主要是下载模型、样本处理函数
# @Reference: https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie/utils.py

import numpy as np
import random

import torch
import os
import sys
import requests
from tqdm import tqdm


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	np.random.seed(seed)


def get_file_from_http(url, temp_file, proxies=None, resume_size=0, user_agent=None):
	"""
	通过请求从 url 下载文件
	:param url: (str)
	:param temp_file:
	:param proxies: (_type_, optional), 默认为 None
	:param resume_size: (int, optional), 默认为 0
	:param user_agent: (_type_, optional), 默认为 None
	"""
	ua = "python/{}".format(sys.version.split()[0])
	ua += "; torch/{}".format(torch.__version__)
	if isinstance(user_agent, dict):
		ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
	elif isinstance(user_agent, str):
		ua += "; " + user_agent
	headers = {"user-agent": ua}

	if resume_size > 0:
		headers["Range"] = "bytes=%d-" % (resume_size,)
	response = requests.get(url, stream=True, proxies=proxies, headers=headers)

	if response.status_code == 416:  # Range not satisfiable
		return

	content_length = response.headers.get("Content-Length")
	total = resume_size + int(content_length) if content_length is not None else None
	progress = tqdm(
		unit="B",
		unit_scale=True,
		total=total,
		initial=resume_size,
		desc="Downloading",
	)
	for chunk in response.iter_content(chunk_size=1024):
		if chunk:  # filter out keep-alive new chunks
			progress.update(len(chunk))
			temp_file.write(chunk)
	progress.close()


def download_pretrained_model(save_path: str, proxies=None):
	"""
	 download config/models with a single url instead of using the Hub
	 在 url 中下载 config/models，而不是用 Hub
	:param save_path: (str) 下载文件的保存地址
	:param proxies: (Dict[str, str], optional) 由协议或端点使用的代理服务器的字典，例如“｛'http'：'foo.bar:3128'，'http://hostname'：'foo.bar:4012'}“ 代理用于每个请求。
	"""
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	base_download_url = 'https://huggingface.co/xyj125/uie-base-chinese/resolve/main'
	files = [
		'pytorch_model.bin',
		'tokenizer.json',
		'tokenizer_config.json',
		'vocab.txt',
		'special_tokens_map.json'
	]

	for tmp_file in files:
		url = os.path.join(base_download_url, tmp_file)
		file = os.path.join(save_path, tmp_file)
		with open(file, "wb") as f:
			get_file_from_http(url, f, proxies=proxies)


if __name__ == '__main__':
	download_pretrained_model('uie-base-chinese')
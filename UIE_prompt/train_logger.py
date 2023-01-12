# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : yingjie.xu
# @File : train_logger.py
# @Description: 训练过程中的记录器，类似于SummaryWriter功能，只是SummaryWriter需要依赖于tensorboard和浏览器做可视化，
# 该工具依赖matplotlib采用静态本地图片存储的形式，便于服务器快速查看训练结果。

import os

import numpy as np
import matplotlib.pyplot as plt


class SummaryWriter(object):
	__doc__ = "训练可视化记录器..."

	def __init__(self, log_path: str, log_name: str, params=[], extention='.png',
	             max_columns=2, log_title=None, figsize=None):
		"""
		创建日志类,初始化函数
		:param log_path: (str) 日志存放文件夹
		:param log_name: (str) 日志文件名
		:param params: (list) 要记录的参数名字列表，如["loss", ...]
		:param extention: (str) 图片存储格式
		:param max_columns: (int) 一行中最多排列几张图
		"""
		self.log_path = log_path
		if not os.path.exists(log_path):
			os.makedirs(log_path)
		self.log_name = log_name
		self.extention = extention
		self.max_param_index = -1
		self.max_columns_threshold = max_columns
		self.figsize = figsize
		self.params_dict = self.create_params_dict(params)
		self.log_title = log_title
		self.init_plt()
		self.update_ax_list()

	def init_plt(self) -> None:
		plt.style.use('seaborn-darkgrid')

	def create_params_dict(self, params: list) -> dict:
		"""
		传入需要记录的变量名列表，创建监控变量字典
		:param params: (list) 监控变量名列表
		:return: (dict) 监控变量名字典 -> {
				'loss': {'values': [0.44, 0.32, ...], 'epochs': [10, 20, ...], 'index': 0},
				'reward': {'values': [10.2, 13.2, ...], 'epochs': [10, 20, ...], 'index': 1},
				...
			}
		"""
		params_dict = {}
		for index, param in enumerate(params):
			params_dict[param] = {'values': [], 'epochs': [], 'index': index}
			self.max_param_index = index
		return params_dict

	def update_ax_list(self) -> None:
		"""
		根据当前的监控变量字典，为每一个变量分配一个图区。
		"""
		# 重新计算每一个变量对应的图幅索引
		params_num = self.max_param_index + 1
		if params_num <= 0:
			return

		self.max_columns = params_num if params_num < self.max_columns_threshold else self.max_columns_threshold
		max_rows = (params_num - 1) // self.max_columns + 1  # 所有变量最多几行
		figsize = self.figsize if self.figsize else (self.max_columns * 6, max_rows * 3)  # 根据图个数计算整个图的 figsize
		self.fig, self.axes = plt.subplots(max_rows, self.max_columns, figsize=figsize)

		# 如果只有一行但又不止一个图，需要手动reshape成(1, n)的形式
		if params_num > 1 and len(self.axes.shape) == 1:
			self.axes = np.expand_dims(self.axes, axis=0)

		# 重新设置 log 标题
		log_title = self.log_title if self.log_title else '[Training Log] {}'.format(self.log_name)
		self.fig.suptitle(log_title, fontsize=15)

	def add_scalar(self, param: str, value: float, epoch: int) -> None:
		"""
		添加一条新的变量值记录
		:param param: (str) 变量名
		:param value: (float) 当前值
		:param epoch: (int) 当前epoch值
		"""
		# 如果该参数是第一次加入，则将该参数加入到监控变量字典中
		if param not in self.params_dict:
			self.max_param_index += 1
			self.params_dict[param] = {'values': [], 'epochs': [], 'index': self.max_param_index}
			self.update_ax_list()

		self.params_dict[param]['values'].append(value)
		self.params_dict[param]['epochs'].append(epoch)

	def record(self, dpi=200) -> None:
		"""
		调用该接口，对该类中目前所有监控的变量状态进行一次记录，将结果保存到本地文件中
		"""
		for param, param_elements in self.params_dict.items():
			param_index = param_elements["index"]
			param_row, param_column = param_index // self.max_columns, param_index % self.max_columns
			ax = self.axes[param_row, param_column] if self.max_param_index > 0 else self.axes
			# ax.set_title(param)
			ax.set_xlabel('Epoch')
			ax.set_ylabel(param)
			ax.plot(self.params_dict[param]['epochs'],
			        self.params_dict[param]['values'],
			        color='darkorange')

		plt.savefig(os.path.join(self.log_path, self.log_name + self.extention), dpi=dpi)


if __name__ == '__main__':
	import random
	import time

	n_epochs = 10
	log_path, log_name = './', 'test'
	writer = SummaryWriter(log_path=log_path, log_name=log_name)
	for i in range(n_epochs):
		loss, reward = 100 - random.random() * i, random.random() * i
		writer.add_scalar('loss', loss, i)
		writer.add_scalar('reward', reward, i)
		writer.add_scalar('random', reward, i)
		writer.record()
		print("Log has been saved at: {}".format(
			os.path.join(log_path, log_name)))
		time.sleep(3)

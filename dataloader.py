from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os, math
import random
import numpy as np
import sys
import h5py,csv
from pathlib import Path
import pdb

class CrackDataset(data.Dataset):
	def __init__(self, img_dir, transform, split=0.7):
		self.img_dir_ = img_dir
		self.train_dataset_ = []
		self.val_dataset_ = []
		self.transform_ = transform
		self.phase_ = 'train'
		self._preprocess(split)

	def make_eval(self):
		self.phase_ = 'eval'

	def make_train(self):
		self.phase_ = 'train'

	def _preprocess(self, split):
		crack_folder = os.path.join(self.img_dir_, 'crack')
		background_folder = os.path.join(self.img_dir_, 'no_crack')
		
		# Add positive examples
		all_crack_files = os.listdir(crack_folder)
		number_of_files = len(all_crack_files)
		
		for i, curr_file in enumerate(all_crack_files):
			full_path = os.path.join(crack_folder, curr_file)
			if os.path.isfile(full_path):
				# Append to training_dataset list
				if i < math.floor(number_of_files * split):
					self.train_dataset_.append([full_path, 1])
				else:
					# Append to validation_data list
					self.val_dataset_.append([full_path, 1])
		
		pos_training_examples = len(self.train_dataset_)
		pos_validation_examples = len(self.val_dataset_)

		# Add negative examples
		background_files = os.listdir(background_folder)
		number_of_files = len(background_files)

		for i, curr_file in enumerate(background_files):
			full_path = os.path.join(background_folder, curr_file)
			if os.path.isfile(full_path):
				# Append to training_dataset
				if i< math.floor(number_of_files * split):
					self.train_dataset_.append([full_path, 0])
				else:
					self.val_dataset_.append([full_path, 0])

		neg_training_examples = len(self.train_dataset_) - pos_training_examples
		neg_validation_examples = len(self.val_dataset_) - pos_validation_examples

		print("Preprocessed dataset...")
		print("Dataset statistics: \n Training examples: {} positive (crack) & {} negative (background).".format(pos_training_examples, neg_training_examples))
		print("Validation examples: {} positive (crack) & {} negative (background).".format(pos_validation_examples, neg_validation_examples))


	def __getitem__(self, index):
		if self.phase_ == 'train':
			# Training
			path, label = self.train_dataset_[index]
		elif self.phase_ == 'eval':
			path, label = self.val_dataset_[index]
		img = Image.open(os.path.join(self.img_dir_, path))
		
		return self.transform_(img), torch.FloatTensor([label])
	
	def __len__(self):
		if self.phase_ == 'train':
			return len(self.train_dataset_)
		elif self.phase_ == 'eval':
			return len(self.val_dataset_)

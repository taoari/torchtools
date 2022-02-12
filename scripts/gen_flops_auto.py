import torch
import timm
import timm.models as models
import numpy as np
import pandas as pd
import os

from torchtools.utils import print_summary

# Flops for all timm models

data = []
missing = []

for i, arch in enumerate(timm.list_models()):
	model = models.__dict__[arch](pretrained=False)
	input_size = model.default_cfg['input_size'] if hasattr(model, 'default_cfg') else (3, 224, 224)
	crop_pct = model.default_cfg['crop_pct'] if hasattr(model, 'default_cfg') and 'crop_pct' in model.default_cfg else 224/256
	input_shape = (1,) + tuple(input_size)
	inputs = torch.randn(*input_shape)
	try:
		res = dict(model=arch, img_size=input_size[-1], crop_pct=crop_pct)
		res.update(print_summary(model, inputs))
		data.append(res)
	except Exception as e:
		missing.append(arch)

	if i % 100 == 0 or i == len(timm.list_models())-1:

		df = pd.DataFrame(data)

		try:
			df['flops_basic'] /= 1e9
			df['flops'] /= 1e9
			df['params'] /= 1e6
			df['params_with_aux'] /= 1e6
		except Exception as e:
			print(e)

		# merge with accuracies if exists
		if os.path.exists('results-imagenet.csv'):
			df_acc = pd.read_csv('results-imagenet.csv').iloc[:,:5]
			# df_acc = df_acc.rename(columns={'model': 'arch'})
			df = df.merge(df_acc, how='left')

		with open('net_arch_auto.md', 'w') as f:
			df.to_markdown(f, index=None)
		with open('missing.txt', 'w') as f:
			f.write('\n'.join(missing))
		print('Missing: {}'.format(missing))
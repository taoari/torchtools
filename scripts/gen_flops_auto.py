import torch
import timm
import timm.models as models
import numpy as np
import pandas as pd

from torchtools.utils import print_summary

# Flops for all timm models

data = []
missing = []

for i, arch in enumerate(timm.list_models()):
	model = models.__dict__[arch](pretrained=False)
	input_size = model.default_cfg['input_size'] if hasattr(model, 'default_cfg') else (3, 224, 224)
	input_shape = (1,) + tuple(input_size)
	inputs = torch.randn(*input_shape)
	try:
		res = dict(arch=arch, input_shape=input_shape)
		res.update(print_summary(model, inputs))
		data.append(res)
	except Exception as e:
		missing.append(arch)

	if (i+1) % 100 == 0 or i == len(timm.list_models())-1:

		df = pd.DataFrame(data)

		try:
			df['flops_basic'] /= 1e9
			df['flops'] /= 1e9
			df['params'] /= 1e6
			df['params_with_aux'] /= 1e6
			df = df.rename(columns={'flops': 'flops (G)', 'flops_basic': 'flops_basic (G)',
				'params': 'params (M)', 'params_with_aux': 'params_with_aux (M)'})
		except Exception as e:
			print(e)

		with open('net_arch_auto.md', 'w') as f:
			df.to_markdown(f, index=None)
		with open('missing.txt', 'w') as f:
			f.write('\n'.join(missing))
		print('Missing: {}'.format(missing))
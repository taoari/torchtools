import torch
import timm
import timm.models as models
import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torchtools.utils import print_summary

# Flops for selected models
ARCHS = [
	'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
	'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
	'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224']

data = []

for arch in ARCHS:
	model = models.__dict__[arch]()
	input_size = model.default_cfg['input_size'] if hasattr(model, 'default_cfg') else (3, 224, 224)
	crop_pct = model.default_cfg['crop_pct'] if hasattr(model, 'default_cfg') else 224/256
	input_shape = (1,) + tuple(input_size)
	inputs = torch.randn(*input_shape)
	res = dict(model=arch, img_size=input_size[-1], crop_pct=crop_pct)
	res.update(print_summary(model, inputs))
	data.append(res)

df = pd.DataFrame(data)

try:
	df['flops_basic'] /= 1e9
	df['flops'] /= 1e9
	df['params'] /= 1e6
	df['params_with_aux'] /= 1e6
	# df = df.rename(columns={'flops_basic': 'flops_basic (G)', 'flops': 'flops (G)',
	# 	'params': 'params (M)', 'params_with_aux': 'params_with_aux (M)'})
except Exception as e:
	print(e)

# merge with accuracies if exists
if os.path.exists('results-imagenet.csv'):
	df_acc = pd.read_csv('results-imagenet.csv').iloc[:,:5]
	# df_acc = df_acc.rename(columns={'model': 'arch'})
	df = df.merge(df_acc, how='left')

with open('net_arch.md', 'w') as f:
	df.to_markdown(f, index=None)

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_context("paper")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

df['cat'] = df['model'].str[:3]
cat2ind = {n:i for i,n in enumerate(sorted(df['cat'].unique()))}
df['color'] = df['cat'].apply(lambda x: colors[cat2ind[x]])

# scatter plot
fig = plt.figure()
ax = None
for c, group in df.groupby(by='color'):
	ax = group.plot(kind='line', x='flops', y='top1', c=c, linestyle='--', ax=ax)
	ax = group.plot(kind='scatter', x='flops', y='top1', c=c, ax=ax)
	# Annotate each data point
	xshift, yshift = 0, 0.25
	for x, y, txt in zip(df['flops'], df['top1'], df['model']):
		ax.annotate(txt, (x+xshift, y+yshift), fontsize=6, ha='right')
ax.get_legend().remove()
plt.tight_layout()
plt.savefig('top1_vs_flops.pdf')


# scatter plot
fig = plt.figure()
ax = None
for c, group in df.groupby(by='color'):
	ax = group.plot(kind='line', x='params', y='top1', c=c, linestyle='--', ax=ax)
	ax = group.plot(kind='scatter', x='params', y='top1', c=c, ax=ax)
	# Annotate each data point
	xshift, yshift = 0, 0.25
	for x, y, txt in zip(df['params'], df['top1'], df['model']):
		ax.annotate(txt, (x+xshift, y+yshift), fontsize=6, ha='right')
ax.get_legend().remove()
plt.tight_layout()
plt.savefig('top1_vs_params.pdf')
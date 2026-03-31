import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from collections import OrderedDict
import torch
import pdb
import os
import umap


def get_tsv(proj):

	label_array = np.array(range(0, 64))
	print(np.shape(proj.tolist()))
	vec_array = np.array(proj.tolist())
	print(vec_array.shape)

	label_df = pd.DataFrame(label_array)
	vec_df = pd.DataFrame(vec_array)

	label_path = "./label.tsv"
	if os.path.exists(label_path):
		os.remove(label_path)
	with open(label_path, 'w') as write_tsv:
		write_tsv.write(label_df.to_csv(sep='\t', index=False, header=False))

	vec_path = "./vector.tsv"
	if os.path.exists(vec_path):
		os.remove(vec_path)
	with open(vec_path, 'w') as write_tsv:
		write_tsv.write(vec_df.to_csv(sep='\t', index=False, header=False))
	print("Finished.")


def train(opt, train_data_loader,val_data_loader , model, visualizer):
	train_dataset = train_data_loader.load_data()
	val_dataset = val_data_loader.load_data()
	train_dataset_size = len(train_data_loader)
	val_dataset_size = len(val_data_loader)
	print('#training images = %d' % train_dataset_size)
	total_steps = 0
	start_iters = 0
	model.ganStep = 0
	model.aeStep = 1

	# 添加内存优化配置
	torch.backends.cudnn.benchmark = True  # 优化卷积运算
	torch.backends.cudnn.deterministic = False  # 允许非确定性算法以获得更好性能

	# 设置内存分配策略
	if hasattr(torch.cuda, 'empty_cache'):
		torch.cuda.empty_cache()
	"""
	print(model.netG.encoder._vq_vae._embedding.weight.data.cpu().shape)
	proj = umap.UMAP(n_neighbors=3,
					 min_dist=0.1,
					 metric='cosine').fit_transform(model.netG.encoder._vq_vae._embedding.weight.data.cpu())
	plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
	plt.savefig('./embeedings.png')
	plt.close()
	get_tsv(model.netG.encoder._vq_vae._embedding.weight.data.cpu())
	"""

	# train
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		# 设置当前epoch，用于自适应损失计算
		model.set_epoch(epoch)
		
		opt.phase = 'train'
		print('#train images = %d' % train_dataset_size)
		model.ganStep = 1
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(train_dataset):
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Restored_Train'],results['Sharp_Train'])
				print('PSNR on Train = %f' %
					  (psnrMetric))
				visualizer.display_current_results(results,epoch)

			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				if opt.display_id > 0:
					visualizer.plot_current_errors(epoch, float(epoch_iter)/train_dataset_size, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				model.save('latest')

		# val
		if epoch % 1 == 0:
			opt.phase = 'val'
			print('start validation')
			print('#val images = %d' % val_dataset_size)
			G_otlatent_val = 0
			G_Content_val = 0
			G_nceloss_val = 0
			for i, data in enumerate(val_dataset):
				model.set_input(data)
				model.validation()
				errors = model.get_current_errors_val()
				G_otlatent_val = G_otlatent_val + errors['G_otlatent_val']
				G_Content_val = G_Content_val + errors['G_Content_val']
				G_nceloss_val = G_nceloss_val + errors['G_nceloss_val']

			errors = OrderedDict([('G_otlatent_val', G_otlatent_val/1000),
						('G_Content_val', G_Content_val/100),
						('G_nceloss_val', G_nceloss_val/1000)
					])
			if opt.display_id > 0:
				visualizer.plot_current_errors_val(epoch, float(epoch_iter)/train_dataset_size, opt, errors)
			print('G_otlatent_val %d ,G_Content_val %d, G_nceloss_val %d' % (G_otlatent_val/1000,G_Content_val/100,G_nceloss_val/1000))
			txtName = "val_loss.txt"
			filedir = os.path.join('./checkpoints/',opt.name,txtName)
			f=open(filedir, "a+")
			recordTime = 'Epoch=' + str(epoch) +'\n'
			new_context = 'G_otlatent_val = '+  str(G_otlatent_val/1000) + ';G_Content_val=' + str(G_Content_val/100) + ';G_latent_L1loss_val=' + str(G_nceloss_val/1000) + '\n'
			f.write(recordTime)
			f.write(new_context)
			torch.cuda.empty_cache()


		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
				  (epoch, total_steps))
			model.save('latest')
			model.save(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate() 


if __name__ == '__main__':
	opt = TrainOptions().parse()
	opt.phase = 'train'
	train_data_loader = CreateDataLoader(opt)
	opt.phase = 'val'
	val_data_loader = CreateDataLoader(opt)
	model = create_model(opt)
	visualizer = Visualizer(opt)
	train(opt, train_data_loader, val_data_loader,model, visualizer)
import time

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
import pdb
import torch
from collections import OrderedDict
import os
import copy  # 添加copy模块导入

def train(opt, data_loader, model, visualizer):

	total_steps = 0
	start_iters = 0
	model.ganStep = 0
	model.aeStep = 1

	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		# === 修复：使用固定基准值计算动态权重 ===
		base_lambda_recon =50.0  # 使用默认值作为基准
		base_lambda_adv = 0.1  # 使用默认值作为基准

		# === 修复：使用基准值而不是opt.lambda_recon/opt.lambda_adv ===
		if epoch < 30:
			# 前期注重重建质量
			current_lambda_recon = base_lambda_recon * 1.5
			current_lambda_adv = base_lambda_adv * 0.8
			#current_lambda_adv = 0
		elif epoch < 60:
			# 中期平衡重建和对抗
			current_lambda_recon = base_lambda_recon
			current_lambda_adv = base_lambda_adv
		else:
			# 后期注重细节生成
			current_lambda_recon = base_lambda_recon * 0.8
			current_lambda_adv = base_lambda_adv * 1.2

		# 更新损失权重
		model.opt.lambda_recon = current_lambda_recon
		model.opt.lambda_adv = current_lambda_adv

		opt.phase = 'train'
		data_loader = CreateDataLoader(opt)
		dataset = data_loader.load_data()
		dataset_size_train = len(data_loader)
		print('#train images = %d' % dataset_size_train)
		model.ganStep = 1
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
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
					visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size_train, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				model.save('latest')

		# === 在每个epoch结束时更新学习率 ===
		for scheduler in schedulers:
			scheduler.step()

		# 打印当前学习率和损失权重
		if schedulers:
			current_lr = schedulers[0].get_last_lr()[0]
			print(f'Epoch {epoch}: LR={current_lr:.6f}, Lambda_recon={current_lambda_recon:.1f}, Lambda_adv={current_lambda_adv:.1f}')
		else:
			print(f'Epoch {epoch}: Lambda_recon={current_lambda_recon:.1f}, Lambda_adv={current_lambda_adv:.1f}')

		# 手动清理数据加载器，释放文件句柄
		if hasattr(data_loader, 'close'):
			data_loader.close()
		torch.cuda.empty_cache()  # 清理GPU缓存

		# start validation
		if epoch % 1 == 0:
			opt.phase = 'val'
			print('start validation')
			data_loader = CreateDataLoader(opt)
			dataset = data_loader.load_data()
			dataset_size = len(data_loader)
			print('#val images = %d' % dataset_size)
			loss_G_Content_val = 0
			loss_G_L1_val = 0
			loss_G_L2_val = 0
			for i, data in enumerate(dataset):
				model.set_input(data)
				model.validation()
				errors = model.get_current_errors_val()
				loss_G_Content_val = loss_G_Content_val+ errors['G_percetual_val']
				loss_G_L1_val = loss_G_L1_val+ errors['G_L1_val']
				loss_G_L2_val = loss_G_L2_val+ errors['G_L2_val']

			errors = OrderedDict([('G_percetual_val', loss_G_Content_val/1000),
						('G_L1_val', loss_G_L1_val/100),
						('G_L2_val', loss_G_L2_val/1000)
					])
			if opt.display_id > 0:
				visualizer.plot_current_errors_val(epoch, float(epoch_iter)/dataset_size_train, opt, errors)
			print('G_percetual_val %d ,G_L1_val %d, G_L2_val %d' % (loss_G_Content_val/1000,loss_G_L1_val/100,loss_G_L2_val/1000))
			txtName = "val_loss.txt"
			filedir = os.path.join('./checkpoints/',opt.name,txtName)
			f=open(filedir, "a+")
			recordTime = 'Epoch=' + str(epoch) +'\n'
			new_context = 'G_percetual_val = '+  str(loss_G_Content_val/1000) + ';G_L1_val=' + str(loss_G_L1_val/100) + ';G_L2_val=' + str(loss_G_L2_val/1000) + '\n'
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


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
train(opt, data_loader, model, visualizer)
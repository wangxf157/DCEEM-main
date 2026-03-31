import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer1 import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR

from util.metrics import SSIM
from util.metrics import ssim
from PIL import Image
import pdb

if __name__ == '__main__':

	opt = TestOptions().parse()	
	opt.nThreads = 1
	opt.batchSize = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip

	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	visualizer = Visualizer(opt)
	# create website
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
	# test
	avgPSNR = 0.0
	avgSSIM = 0.0
	counter = 0

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
		avgPSNR += PSNR(visuals['fake_B'],visuals['real_B'])

		avgSSIM += ssim(visuals['fake_B'],visuals['real_B'])
		# 如果是VQGAN模型，添加额外的评估指标
		if hasattr(model, 'vq_loss') and hasattr(model, 'perplexity'):
			avgVQLoss += model.vq_loss.item()
			avgPerplexity += model.perplexity.item()

		img_path = model.get_image_paths()
		print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)

	
	avgPSNR /= counter
	avgSSIM /= counter
	txtName = "note.txt"
	filedir = os.path.join(web_dir,txtName)
	f=open(filedir, "a+")

	if hasattr(model, 'vq_loss') and hasattr(model, 'perplexity'):
		avgVQLoss /= counter
		avgPerplexity /= counter
		new_context = 'PSNR = {:.4f}; SSIM = {:.4f}; VQ_Loss = {:.4f}; Perplexity = {:.4f}\n'.format(
			avgPSNR, avgSSIM, avgVQLoss, avgPerplexity
		)
		print('PSNR = {:.4f}, SSIM = {:.4f}, VQ_Loss = {:.4f}, Perplexity = {:.4f}'.format(
			avgPSNR, avgSSIM, avgVQLoss, avgPerplexity))
	else:
		new_context = 'PSNR = {:.4f}; SSIM = {:.4f}\n'.format(avgPSNR, avgSSIM)
		print('PSNR = {:.4f}, SSIM = {:.4f}'.format(avgPSNR, avgSSIM))

	f.write(new_context)
	webpage.save()
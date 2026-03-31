"""
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import getpsnr
from util.metrics import SSIM
from util.metrics import ssim
from PIL import Image
import cv2


if __name__ == '__main__':

	opt = TestOptions().parse()
	opt.nThreads = 1
	opt.batchSize = 1
	opt.serial_batches = True
	opt.no_flip = True

	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)

	# create website
	# test
	avgPSNR = 0.0
	avgSSIM = 0.0
	avgPSNR_i = 0.0
	avgSSIM_i = 0.0
	avgPSNR_1 = 0.0
	counter = 0

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		# pdb.set_trace()
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
		avgPSNR += PSNR(visuals['fake_B'],visuals['real_B'])
		avgPSNR_i += PSNR(visuals['fake_Bi'],visuals['real_B'])
		avgPSNR_1 += getpsnr(visuals['fake_Bi'],visuals['real_B'])
		avgSSIM += ssim(visuals['fake_B'],visuals['real_B'])
		avgSSIM_i += ssim(visuals['fake_Bi'],visuals['real_B'])
		img_path = model.get_image_paths()[0]
		s = img_path.rsplit('\\', 2)
		img_path = s[0]+'\\restored_attn\\'+s[2]
		print('process image... %s' % img_path)
		#print(img_path)
		
		image = cv2.cvtColor(visuals['fake_Bi'], cv2.COLOR_BGR2RGB)
		print(img_path)
		cv2.imwrite(img_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

		image = cv2.cvtColor(visuals['real_B'], cv2.COLOR_BGR2RGB)
		cv2.imwrite(s[0]+'\\restored_attn\\2_'+s[2], image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	
	
	avgPSNR /= counter
	avgSSIM /= counter
	avgPSNR_i /= counter
	avgPSNR_1 /= counter
	avgSSIM_i /= counter

	print('PSNR = %f, SSIM = %f,PSNR_i = %f, PSNR_1 = %f, SSIM_i = %f' %
					(avgPSNR, avgSSIM, avgPSNR_i,avgPSNR_1, avgSSIM_i))
"""

import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
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
		avgPSNR += PSNR(visuals['fake_Bi'],visuals['real_B'])

		avgSSIM += ssim(visuals['fake_Bi'],visuals['real_B'])
		img_path = model.get_image_paths()
		print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)

		
	avgPSNR /= counter
	avgSSIM /= counter
	txtName = "note.txt"
	filedir = os.path.join(web_dir,txtName)
	f=open(filedir, "a+")
	new_context = 'PSNR = '+  str(avgPSNR) + ';SSIM=' + str(avgSSIM) + '\n'
	f.write(new_context)
	print('PSNR = %f, SSIM = %f' %
					(avgPSNR, avgSSIM))

	webpage.save()

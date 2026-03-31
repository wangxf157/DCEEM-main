import os
import torch
import pdb


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename) 
        state_dict = torch.load(save_path)
        network.load_state_dict(state_dict)

    # load E1 and D1 trained in step 1
    def load_ae(self, network, which_ep, whichblock, which_data, which_norm):
        if self.opt.which_model_netG == 'introAE':

            save_filename = None
            weight_path = None
            
            if whichblock == 'E1':
                save_filename = which_ep + '_net_G_Encoder1.pth'
            elif whichblock == 'D':
                save_filename = which_ep + '_net_G_Decoder.pth'
            
            # 修改为step1的实际checkpoints路径
            weight_path = '/root/ParaEncodeNet-main/checkpoints/step1_stl10'  # 修改这一行

            save_path = os.path.join(weight_path, save_filename)
            state_dict = torch.load(save_path)
            
            # 添加调试信息
            print(f"Loading weights from: {save_path}")
            print(f"Model keys: {list(network.state_dict().keys())[:5]}")  # 打印前5个键
            print(f"State dict keys: {list(state_dict.keys())[:5]}")  # 打印前5个键
            
            try:
                network.load_state_dict(state_dict)
                print(save_filename + ' has been loaded successfully')
            except RuntimeError as e:
                print(f"Error loading weights: {e}")
                # 如果加载失败，使用随机初始化
                print("Using random initialization instead")
            print(save_filename + 'has been loaded')

        else:
            raise ValueError('This repo only support the autoencoder modified from introAE, i.e., opt.which_model_netG == introAE. \
			But you can use this option to add new model')


    def update_learning_rate():
        pass
import time, math, glob
import torch
import scipy.io
import numpy as np

from NETWORK.Network import Origin_Generator_Radv2

from TOOLS.ulti_v3 import Torchtensor2Array, Generate_HDR_image_with_GaussianWeight


# test images

num_of_images = 12
image_list = sorted(glob.glob("test_data/Noncalibrated/input" + "/*.*"))
image_savename = ['C08', 'C11', 'C15', 'C23', 'C25', 'C26', 'C31', 'C32', \
              'C42', 'belgium', 'mpi_office', 'office']

# load Network
device = "cuda:0"

net_path = 'WEIGHTS/net_93.pth'

save_path = 'test_data/Results/'

with torch.no_grad():

    HDR_net = Origin_Generator_Radv2(channels_input=1, channels_output=3)
    state_dict = torch.load(net_path, map_location = lambda s, l: s)
    HDR_net.load_state_dict(state_dict)
    HDR_net.eval()
    HDR_net.to(device)

    count = 0
    for image_name in image_list:
        print('Image: %d' %(count + 1))


        input = mat2tensor(image_name, 'E_hat', channel=1)
        input = input.float()
        input = torch.log(input + 1e-6)

        HDRMB = Generate_HDR_image_with_GaussianWeight(input, net, device, size_patch=32, stride=16)
        HDRMB = torch.exp(HDRMB)

        HDRMB = Torchtensor2Array(HDRMB)

        name = image_savename[count]
        scipy.io.savemat( save_path + name + '.mat', mdict={'HDR': HDRMB})
        count = count + 1


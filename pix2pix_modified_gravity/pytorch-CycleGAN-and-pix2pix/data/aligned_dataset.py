import os
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.base_dataset import custom_density_forward
from data.image_folder import make_dataset
# from PIL import Image
import torch

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path)  # EDIT: Changed
        # split AB image into A and B
        _, w = AB.shape
        w2 = int(w / 2)
        A = AB[:, :w2]
        B = AB[:, w2:]

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.shape[::-1])
        A_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False)

        A = torch.from_numpy(A.astype(np.float32)).unsqueeze(0)
        B = torch.from_numpy(B.astype(np.float32)).unsqueeze(0)

        ############# USE BELOW LINES FOR DENSITY ##############
        B = B / A

        A = torch.log(A)/10
        B = torch.log(B)/10
        ########################################################

        ##### For F4-veldiv #####
        #A = torch.from_numpy(A).to(torch.double).unsqueeze(0)
        #B = torch.from_numpy(B).to(torch.double).unsqueeze(0)
        #B = (B + 16000)/(A + 16000)
        #A = torch.log(A+16000)/11  # float64 is used to prevent any mismatch errors during untransformation.
        #B = torch.log(B)/4
        #########################

        # Extra#########################################
        #A = torch.log(A)/torch.log(torch.tensor([4000]))
        #B = torch.log(B)/torch.log(torch.tensor([900]))

        #A[A >= 1] = 1
        #B[B >= 1] = 1
        ################################################

        #A = boxcox(A, lmbda=-0.3458325086781785)/12
        #B = boxcox(B, lmbda=-0.3458325086781785)/60

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
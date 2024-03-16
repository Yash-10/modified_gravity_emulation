import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
#from PIL import Image
import numpy as np
import torch

def andres_forward(x, a=4):
    return 2 * x / (x + a) - 1


def veldiv_forward(img, scale1=-700, scale2=200):
    img[img > scale2] = scale2
    img[img < scale1] = scale1

    img[img < 0] /= np.abs(scale1)
    img[img > 0] /= np.abs(scale2)

    return img

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
        #AB = Image.open(AB_path).convert('RGB')
        AB = np.load(AB_path)
        # split AB image into A and B
        # split AB image into A and B
        _, w = AB.shape
        w2 = int(w / 2)
        A = AB[:, :w2]
        B = AB[:, w2:]

        #w, h = AB.size
        #w2 = int(w / 2)
        #A = AB.crop((0, 0, w2, h))
        #B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        transform_params = get_params(self.opt, A.shape[::-1])
        A_transform = get_transform(self.opt, transform_params, grayscale=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=False)

        A = torch.from_numpy(A.astype(np.float64)).unsqueeze(0)
        B = torch.from_numpy(B.astype(np.float64)).unsqueeze(0)

        #B = B / A

        #A = torch.log(A)/10
        #B = torch.log(B)/10
        ########## Below lines for density. ##########
        #A = andres_forward(A, a=7)
        #B = andres_forward(B, a=7)
        ##############################################
        #### Below lines for velocity divergence #####
        A = veldiv_forward(A, scale1=-900, scale2=400)
        B = veldiv_forward(B, scale1=-900, scale2=400)
        ##############################################

        A, B = A.float(), B.float()

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

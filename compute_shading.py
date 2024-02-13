import imageio
import albedolib
import numpy as np
#import matplotlib.pyplot as plt
import cv2
import scipy
#import torch
import utils
import argparse
import multiprocessing as mp
import math
import rawpy_convert
import os
import gc

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
def ksize_from_sigma(sigma):                                                                                                                                   
    num= int(math.ceil(-(20/3) * (0.35 - sigma)))
    if num % 2 == 0:
        num += 1
        pass
    return num

def region_wise_blur(I, M, sigma):
    ksize = ksize_from_sigma(sigma)
    I_blur = I.copy()
    gaussian = gkern(ksize, sigma).astype(I.dtype)
    ones = np.ones((ksize, ksize))
    ms = np.unique(M)
    for m in ms[1:]:
        I_patch = I.copy()
        #print()
        mask = M==m
        I_patch[~mask] = 0
        weight_init = mask.astype(I.dtype)
        #print(I_patch.shape)
        #I_blur_torch = torch.tensor(I.copy()).permute(2,0,1).unsqueeze(0)
        #print(I_blur_torch.dtype)
        #for i in range(3):
        #    print(I_blur_torch[:, i,i+1].shape)
        #    I_blur_torch[:, i:i+1] = conv(I_blur_torch[:, i:i+1])
        #    pass
        #I_blur = I_blur_torch.numpy().squeeze(0).transpose(1,2,0)
        #plt.imshow(I_blur)
        I_patch_blur = cv2.filter2D(I_patch, ddepth=-1, kernel=gaussian)
        # print("i_patch_blur", I_patch_blur.dtype)
        weight = cv2.filter2D(weight_init, ddepth=-1, kernel=gaussian)
        # print("weight", weight.dtype)
        # print("I_blur", I_blur.dtype)
        #plt.figure()
        #plt.imshow(I_patch_blur)
        #plt.figure()
        #plt.imshow(weight)
        #print(weight.min(), weight.max())
        #print(weight)
        result = np.nan_to_num(I_patch_blur/weight[:,:,np.newaxis], posinf=0, neginf=0)
        #print(result.min(), result.max())
        #print(I_patch_blur.min(), I_patch_blur.max())
        #plt.figure()
        #plt.imshow(result)
        I_blur[mask] = result[mask]
        pass
    return I_blur

def masked_blur(I, M, sigma): #TODO
    ksize = ksize_from_sigma(sigma)
    I_blur = I.copy()
    gaussian = gkern(ksize, sigma).astype(I.dtype)    
    mask = M != 0
    I_blur[~mask] = 0
    
    weight_init = mask.astype(I_blur.dtype)
    I_blur_raw = cv2.filter2D(I_blur, ddepth=-1, kernel=gaussian)
    weight = cv2.filter2D(weight_init, ddepth=-1, kernel=gaussian)
    I_blur = np.nan_to_num(I_blur_raw/weight[:,:,np.newaxis], posinf=0, neginf=0)
    I_blur[~mask] = 0
    return I_blur

def naive_blur(I, M, sigma):
    ksize = ksize_from_sigma(sigma)
    I_blur = I.copy()
    gaussian = gkern(ksize, sigma).astype(I.dtype)    
    mask = M != 0
    I_blur[~mask] = 0
    
    I_blur = cv2.filter2D(I_blur, ddepth=-1, kernel=gaussian)
    return I_blur

def compute_shading(I_l, C, M):

    
    #I_l = albedolib.srgb2lin(I/255)
    #A_l = albedolib.srgb2lin(A/255)

    I_l = albedolib.srgb2lin(albedolib.adobelin2srgb(I_l)).astype(np.float32)

    A_l = np.zeros((M.shape[0], M.shape[1], 3), dtype=I_l.dtype) 
    indices = np.unique(M)
    for i, idx in enumerate(indices[1:]):
        A_l[M == idx] = albedolib.srgb2lin(albedolib.adobelin2srgb(C[i]))
        pass
    #print(gaussian.shape)
    #conv = partialconv.models.partialconv2d.PartialConv2d(1, 1, 124, stride=1, padding=0, bias=False)
    #print(conv.weight.shape)
    #conv.weight = torch.nn.Parameter(torch.tensor(gaussian[np.newaxis, np.newaxis]))
    
    #I_blur_l = albedolib.srgb2lin(I_blur/255)
    # if blur_type == "region_wise":
    #     I_blur_l = region_wise_blur(I_l, M, sigma=20)
    #     pass
    # elif blur_type == "masked":
    #     I_blur_l = masked_blur(I_l, M, sigma=20)
    #     pass
    # else:
    #     raise NotImplementedError("blur type: {}".format(blur_type))
    I_blur_l = region_wise_blur(I_l, M, sigma=20)
    # print("I_blur_l", I_blur_l.dtype)
    S_l = np.nan_to_num(I_l / A_l, posinf=0, neginf=0) 
    S_blur_l = np.nan_to_num(I_blur_l / A_l, posinf=0, neginf=0) 
    max_all = np.maximum(S_l.max(), S_blur_l.max())
    
    S_vis = albedolib.lin2srgb(S_l / max_all) 
    S_blur_vis = albedolib.lin2srgb(S_blur_l / max_all)
    
    return S_l, S_blur_l, S_vis, S_blur_vis

def compute_shading_ver2(I_l, C, M):    
    #I_l = albedolib.srgb2lin(I/255)
    #A_l = albedolib.srgb2lin(A/255)

    I_l = albedolib.srgb2lin(albedolib.adobelin2srgb(I_l)).astype(np.float32)

    A_l = np.zeros((M.shape[0], M.shape[1], 3), dtype=I_l.dtype) 
    indices = np.unique(M)
    for i, idx in enumerate(indices[1:]):
        A_l[M == idx] = albedolib.srgb2lin(albedolib.adobelin2srgb(C[i]))
        pass
    S_l = np.nan_to_num(I_l / A_l, posinf=0, neginf=0) 
    S_l[M==0] = 0
    S_blur_l = masked_blur(S_l, M, sigma=20)
    # S_blur_l = region_wise_blur(S_l, M, sigma=20)
    max_all = np.maximum(S_l.max(), S_blur_l.max())
    
    S_vis = albedolib.lin2srgb(S_l / max_all) 
    S_blur_vis = albedolib.lin2srgb(S_blur_l / max_all)
    
    return S_l, S_blur_l, S_vis, S_blur_vis

def compute_shading_ver3(I_l, C, M):    
    I_l = albedolib.srgb2lin(albedolib.adobelin2srgb(I_l)).astype(np.float32)
    A_l = np.zeros((M.shape[0], M.shape[1], 3), dtype=I_l.dtype) 
    indices = np.unique(M)
    for i, idx in enumerate(indices[1:]):
        A_l[M == idx] = albedolib.srgb2lin(albedolib.adobelin2srgb(C[i]))
        pass
    S_l = np.nan_to_num(I_l / A_l, posinf=0, neginf=0) 
    S_l[M==0] = 0
    # S_blur_l = masked_blur(S_l, M, sigma=20)
    # S_blur_l = region_wise_blur(S_l, M, sigma=20)
    S_blur_l = naive_blur(S_l, M, sigma=20)
    max_all = np.maximum(S_l.max(), S_blur_l.max())
    
    S_vis = albedolib.lin2srgb(S_l / max_all) 
    S_blur_vis = albedolib.lin2srgb(S_blur_l / max_all)
    
    return S_l, S_blur_l, S_vis, S_blur_vis


def load_save_shading(row):
    gc.collect()
    image = row.get_img_path()
    print(image)
    albedo = row.get_gt_albedo()
    mask = row.get_mask()
    color_lib = row.get_color_lib()
    I_raw_path = image.replace("_png", "_raw")
    I_raw_path_sony = I_raw_path.replace(".png", ".ARW")
    I_raw_path_nikon = I_raw_path.replace(".png", ".NEF")
    if os.path.exists(I_raw_path_sony):
        # print("I_raw_path_sony", I_raw_path_sony)
        I_raw = rawpy_convert.process_img(I_raw_path_sony, mode="full")
        pass
    else:
        # print("I_raw_path_nikon", I_raw_path_nikon)
        I_raw = rawpy_convert.process_img(I_raw_path_nikon, mode="full")
        pass
    I_raw = I_raw.astype(np.float32) / 65535
    #I = imageio.imread(image)
    C = np.load(color_lib)
    M = imageio.imread(mask)
        
    s_l, s_blur_l, s_vis, s_blur_vis = compute_shading_ver3(I_raw, C, M)
    # print(s_l.max())
    shading_path = albedo.replace("_albedo.png", "_shading.png")
    shading_npy_path = albedo.replace("_albedo.png", "_shading.npy")
    
    shading_blur_path = albedo.replace("_albedo.png", "_shading_blur.png")
    shading_blur_npy_path = albedo.replace("_albedo.png", "_shading_blur.npy")
    
    print("shading", shading_path)
    imageio.imwrite(shading_path, s_vis)
    imageio.imwrite(shading_blur_path, s_blur_vis)
    np.savez_compressed(shading_npy_path, s=s_l.astype(np.float32))
    np.savez_compressed(shading_blur_npy_path, s=s_blur_l.astype(np.float32))
    pass

def main():
    # gc.set_debug(gc.DEBUG_STATS)
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--imgs_dir", type=str, default=None)
    args = parser.parse_args()
    
    finder = utils.PathFinderMgr(args.meta_path, imgs_dir=args.imgs_dir)
    examples = [ex for ex in list(finder.iter_examples()) if "_DSC3428" in ex.get_img_path()]
    #examples = list(finder.iter_examples())
    with mp.Pool() as p:
        p.map(load_save_shading, examples)
    # print(len(examples))
    # list(map(load_save_shading, examples))
# #   
#         pass

    pass

if __name__ == "__main__":
    main()
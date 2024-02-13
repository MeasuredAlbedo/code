import numpy as np
import imageio
import cv2
import PIL.Image

import os
import utils
import csv
from html4vision import Col, imagetable
import skimage.measure
import matplotlib.pyplot as plt
import albedolib
#import pytorch3d.loss
import torch
import largestinteriorrectangle as lir
import lpips
import torch
import argparse
import math
def gaussian_highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 0.5


class TextureEvaluator:
    def __init__(self):
        pass
    def high_freq_from_path(self, img_path):
        img = PIL.Image.open(img_path)
        img_low = img.resize((640, 480), resample=PIL.Image.BICUBIC)
        img_low = np.array(img_low, dtype=np.float32)
        highpass = gaussian_highpass(img_low, 1)
        return highpass
    
    def evaluate(self, albedo_path, img_path, shading_mask_path):
        albedo_highpass = self.high_freq_from_path(albedo_path)
        img_highpass = self.high_freq_from_path(img_path)
        err = np.linalg.norm(albedo_highpass - img_highpass)
        return err 
    pass

def count_edges(albedo_edges, img_resized_edges, smooth_label_resized, debug=False):
    tp_img = np.logical_and(img_resized_edges, albedo_edges)
    tp_mask = tp_img[smooth_label_resized]
    tp_num = np.count_nonzero(tp_mask)

    tn_img = np.logical_and(np.logical_not(img_resized_edges), np.logical_not(albedo_edges))

    fp_img = np.logical_and(np.logical_not(img_resized_edges), albedo_edges)
    fp_mask = fp_img[smooth_label_resized]
    fp_num = np.count_nonzero(fp_mask)

    fn_img = np.logical_and(img_resized_edges, np.logical_not(albedo_edges))
    fn_mask = fn_img[smooth_label_resized]
    fn_num = np.count_nonzero(fn_mask)
    debug = {
        "tp_img": tp_img,
        "fp_img": fp_img,
        "fn_img": fn_img
    }
    if not debug:
        return tp_num, fp_num, fn_num
    else:
        return tp_num, fp_num, fn_num, debug
def score_edges(albedo_edges, img_resized_edges, smooth_label_resized):
    tp_num, fp_num, fn_num = count_edges(albedo_edges, img_resized_edges, smooth_label_resized)
    recall = tp_num / (tp_num + fn_num)
    precision = tp_num / (tp_num + fp_num) if tp_num+fp_num != 0 else 0
    f1 = 2* (precision * recall) / (precision + recall) if (precision+recall) != 0 else 0
    return recall, precision, f1
    
class TextureEdgeEvaluator:
    def __init__(self):
        pass
    
    def evaluate(self, img_path, label_path, albedo_path):
        img = np.array(PIL.Image.open(img_path).convert('L'))
        img_edges = cv2.Canny(img, 25, 25)
        print(img_path)
        label = np.load(label_path)
        smooth_label = label == 2
        
        img_edges_vis = np.tile(np.zeros_like(img_edges)[:, :, np.newaxis], [1, 1, 3])
        img_edges_vis[smooth_label] = np.array([255,0,0])
        img_edges_vis[img_edges == 255] = 255
        imageio.imwrite(os.path.join("vis_edges", os.path.basename(img_path)), img_edges_vis)

        albedo = np.array(PIL.Image.open(albedo_path).convert('L'))
        img_resized = np.array(PIL.Image.fromarray(img).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.LANCZOS))
        img_resized_edges = cv2.Canny(img_resized, 25, 25)
        smooth_label_resized = np.array(PIL.Image.fromarray(smooth_label).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.NEAREST))
        
        albedo_edges_sweep = [cv2.Canny(albedo, i, i) for i in range(1, 255)]
        albedo_edges_scores = [score_edges(albedo_edges, img_resized_edges, smooth_label_resized) for albedo_edges in albedo_edges_sweep]
        max_score_idx, (recall, precision, f1) =  max(enumerate(albedo_edges_scores), key=lambda x:x[1][2])
        albedo_edges = albedo_edges_sweep[max_score_idx]
        
        albedo_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_edges.png")
        albedo_gray_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_gray.png")
        img_resized_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_resized.png")
        img_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_edges.png")
        smooth_label_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_smooth_label.png")
        imageio.imwrite(albedo_gray_path, albedo)
        imageio.imwrite(albedo_edges_path, albedo_edges)
        imageio.imwrite(img_resized_path, img_resized)
        imageio.imwrite(img_edges_path, img_resized_edges)
        imageio.imwrite(smooth_label_path, smooth_label_resized.astype(np.uint8)*255)

        print(albedo_path)
        print('recall', recall)
        print('precision', precision)
        print('f1', f1)
        return recall, precision, f1
        pass
    pass

class TextureEdgeEvaluatorV2:
    def __init__(self):
        pass
    
    def evaluate(self, img_path, label_path, albedo_path):
        print(img_path)
        img = np.array(PIL.Image.open(img_path))
        label = np.load(label_path)
        smooth_label = label == 2

        img_edges = cv2.Canny(img, 40, 40)
        img_edges_vis = np.tile(np.zeros_like(img_edges)[:, :, np.newaxis], [1, 1, 3])
        img_edges_vis[smooth_label] = np.array([255,0,0])
        img_edges_vis[img_edges == 255] = 255
        imageio.imwrite(os.path.join("vis_edges", os.path.basename(img_path)), img_edges_vis)

        
        albedo = np.array(PIL.Image.open(albedo_path))
        albedo_linear = albedolib.srgb_to_linsrgb(albedo / 255) 
        albedo_linear_gray = albedo_linear.mean(axis=-1)
        
        img_resized = np.array(PIL.Image.fromarray(img).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.LANCZOS))
        img_resized_linear = albedolib.srgb_to_linsrgb(img_resized / 255) 
        img_resized_linear_gray = img_resized_linear.mean(axis=-1)
        
        smooth_label_resized = np.array(PIL.Image.fromarray(smooth_label).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.NEAREST))
        
        labels = skimage.measure.label(smooth_label_resized)
        indices = np.unique(labels)

        tp_sum = 0
        fp_sum = 0
        fn_sum = 0

        img_edges_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        albedo_edges_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        img_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        albedo_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)

        tp_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        fp_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        fn_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        def calc_metrics(tp_sum, fp_sum, fn_sum):
            recall = tp_sum / (tp_sum + fn_sum) if tp_sum+fn_sum != 0 else 0    
            precision = (tp_sum / (tp_sum + fp_sum)) if (tp_sum+fp_sum) != 0 else 0
            f1 = 2* (precision * recall) / (precision + recall) if (precision+recall) != 0 else 0
            return recall, precision, f1
        img_has_valid_patch = False
        for idx in indices[1:]:
            region_mask = labels == idx

            img_resized_linear_region_avg = img_resized_linear_gray[region_mask].mean()
            img_resized_linear_rescaled = (img_resized_linear_gray / img_resized_linear_region_avg * 0.21404114)

            img_resized_rescaled = (albedolib.lin2srgb(img_resized_linear_rescaled)*255).round().astype(np.uint8)

            img_resized_rescaled_edges = cv2.Canny(img_resized_rescaled, 45, 45)
            
            print("albedopath", albedo_path, img_resized_rescaled_edges[region_mask].mean() / 255, img_resized_rescaled_edges[region_mask].sum() / 255)
            if  img_resized_rescaled_edges[region_mask].sum() / 255 < 20:
                continue

            albedo_linear_region_avg = albedo_linear_gray[region_mask].mean()
            albedo_linear_rescaled = (albedo_linear_gray / albedo_linear_region_avg * 0.21404114 )
            
            albedo_rescaled = (albedolib.lin2srgb(albedo_linear_rescaled) * 255).round().astype(np.uint8)
            
            albedo_vis[region_mask] = albedo_rescaled[region_mask]
            img_vis[region_mask] = img_resized_rescaled[region_mask]
            
            img_has_valid_patch = True
            def sweep(threshold):
                albedo_rescaled_edges = cv2.Canny(albedo_rescaled, threshold, threshold)    
                tp_num, fp_num, fn_num, debug = count_edges(albedo_rescaled_edges, img_resized_rescaled_edges, region_mask, debug=True)
                return {"albedo_edges": albedo_rescaled_edges,
                        "img_edges": img_resized_rescaled_edges,
                        "tp_num": tp_num,
                        "fp_num": fp_num,
                        "fn_num": fn_num,
                        "debug":debug
                        }
            
            results = [(threshold, sweep(threshold)) for threshold in range(30, 255)]
            def res_cmp(threshold_result):
                tp, fp, fn = threshold_result[1]["tp_num"], threshold_result[1]["fp_num"], threshold_result[1]["fn_num"]
                recall,precision,f1=calc_metrics(tp, fp, fn)
                return (f1, recall, precision, tp, -fp, -fn)
            
            threshold, result = max(results, key=lambda threshold_result: res_cmp(threshold_result))
            print("albedo_path", albedo_path, "threshold", threshold)
            
            albedo_rescaled_edges = result["albedo_edges"]
            img_resized_rescaled_edges = result["img_edges"]
            tp_num, fp_num, fn_num, debug = result["tp_num"], result["fp_num"], result["fn_num"], result["debug"]
            albedo_edges_vis[region_mask] = albedo_rescaled_edges[region_mask]
            img_edges_vis[region_mask] = img_resized_rescaled_edges[region_mask]
            print("tp_num", tp_num, "fp_num", fp_num, "fn_num", fn_num)
            tp_sum += tp_num
            fp_sum += fp_num
            fn_sum += fn_num
            tp_vis[region_mask] = np.logical_or(tp_vis[region_mask], debug["tp_img"][region_mask])
            fp_vis[region_mask] = np.logical_or(fp_vis[region_mask], debug["fp_img"][region_mask])
            fn_vis[region_mask] = np.logical_or(fn_vis[region_mask], debug["fn_img"][region_mask])
            #count_edges(albedo_rescaled)
            pass
        
        #print("final albedo_path", albedo_path, "tp", tp_sum, "fp", fp_sum, "fn", fn_sum)        
        if not img_has_valid_patch:
            return None
        
        recall, precision, f1= calc_metrics(tp_sum, fp_sum, fn_sum)
        
        albedo_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_edges.png")
        albedo_gray_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_gray.png")
        img_resized_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_resized.png")
        img_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_edges.png")
        smooth_label_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_smooth_label.png")
        smooth_label_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_smooth_label.png")
        tp_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_tp.png")
        fp_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_fp.png")
        fn_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_fn.png")
        
        imageio.imwrite(albedo_gray_path, albedo_vis)
        imageio.imwrite(albedo_edges_path, albedo_edges_vis)
        imageio.imwrite(img_resized_path, img_vis)
        imageio.imwrite(img_edges_path, img_edges_vis)
        imageio.imwrite(smooth_label_path, smooth_label_resized.astype(np.uint8)*255)
        imageio.imwrite(tp_path, tp_vis*255)
        imageio.imwrite(fp_path, tp_vis*255)
        imageio.imwrite(fn_path, tp_vis*255)
        
        return recall, precision, f1
        pass
    pass
class TextureEdgeEvaluatorV3:
    def __init__(self):
        pass
    
    def evaluate(self, img_path, label_path, albedo_path):
        print(img_path)
        img = np.array(PIL.Image.open(img_path))
        label = np.load(label_path)
        smooth_label = label == 2

        img_edges = cv2.Canny(img, 25, 25)
        img_edges_vis = np.tile(np.zeros_like(img_edges)[:, :, np.newaxis], [1, 1, 3])
        img_edges_vis[smooth_label] = np.array([255,0,0])
        img_edges_vis[img_edges == 255] = 255
        imageio.imwrite(os.path.join("vis_edges", os.path.basename(img_path)), img_edges_vis)

        
        albedo = np.array(PIL.Image.open(albedo_path))
        albedo_linear = albedolib.srgb_to_linsrgb(albedo) / 255
        albedo_linear_gray = albedo_linear.mean(axis=-1)
        img_resized = np.array(PIL.Image.fromarray(img).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.LANCZOS))
        img_resized_linear = albedolib.srgb_to_linsrgb(img_resized) / 255
        img_resized_linear_gray = img_resized_linear.mean(axis=-1)
        
        smooth_label_resized = np.array(PIL.Image.fromarray(smooth_label).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.NEAREST))
        
        labels = skimage.measure.label(smooth_label_resized)
        #plt.imshow(labels)
        #plt.show()
        #exit()
        indices = np.unique(labels)

        tp_sum = 0
        fp_sum = 0
        fn_sum = 0

        img_edges_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        albedo_edges_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        img_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        albedo_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        img_size = np.array([albedo.shape[0], albedo.shape[1]])
        tp_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        fp_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        fn_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        def calc_metrics(tp_sum, fp_sum, fn_sum):
            recall = tp_sum / (tp_sum + fn_sum) if tp_sum+fn_sum != 0 else 0    
            precision = (tp_sum / (tp_sum + fp_sum)) if (tp_sum+fp_sum) != 0 else 0
            f1 = 2* (precision * recall) / (precision + recall) if (precision+recall) != 0 else 0
            return recall, precision, f1

        for idx in indices[1:]:
            region_mask = labels == idx

            albedo_linear_region_avg = albedo_linear_gray[region_mask].mean()
            albedo_linear_rescaled = (albedo_linear_gray / albedo_linear_region_avg * 0.21404114 )

            img_resized_linear_region_avg = img_resized_linear_gray[region_mask].mean()
            img_resized_linear_rescaled = (img_resized_linear_gray / img_resized_linear_region_avg * 0.21404114)

            albedo_rescaled = (albedolib.lin2srgb(albedo_linear_rescaled) * 255).round().astype(np.uint8)
            img_resized_rescaled = (albedolib.lin2srgb(img_resized_linear_rescaled)*255).round().astype(np.uint8)

            albedo_vis[region_mask] = albedo_rescaled[region_mask]
            img_vis[region_mask] = img_resized_rescaled[region_mask]
            img_resized_rescaled_edges = cv2.Canny(img_resized_rescaled, 50, 50)

            img_resized_rescaled_edges_masked = img_resized_rescaled_edges.copy()
            img_resized_rescaled_edges_masked[~region_mask] = 0
            
            img_pos = np.array(np.nonzero(img_resized_rescaled_edges_masked)).transpose()
            img_pos_norm = img_pos / img_size
            
            def sweep(threshold):
                albedo_rescaled_edges = cv2.Canny(albedo_rescaled, threshold, threshold)
                albedo_rescaled_edges_masked = albedo_rescaled_edges.copy()
                albedo_rescaled_edges_masked[~region_mask] = 0
                
                albedo_pos = np.array(np.nonzero(albedo_rescaled_edges_masked)).transpose()
                albedo_pos_norm = albedo_pos / img_size
                
                patch_dist = pytorch3d.loss.chamfer_distance(torch.tensor(img_pos_norm, dtype=torch.float).unsqueeze(0), torch.tensor(albedo_pos_norm, dtype=torch.float).unsqueeze(0))

                return {"albedo_edges": albedo_rescaled_edges,
                        "img_edges": img_resized_rescaled_edges,
                        "tp_num": tp_num,
                        "fp_num": fp_num,
                        "fn_num": fn_num,
                        "debug":debug
                        }
            
            results = [(threshold, sweep(threshold)) for threshold in range(35, 255)]
            def res_cmp(threshold_result):
                tp, fp, fn = threshold_result[1]["tp_num"], threshold_result[1]["fp_num"], threshold_result[1]["fn_num"]
                recall,precision,f1=calc_metrics(tp, fp, fn)
                return (f1, recall, precision, tp, -fp, -fn)
            
            threshold, result = max(results, key=lambda threshold_result: res_cmp(threshold_result))
            print("albedo_path", albedo_path, "threshold", threshold)
            # for res in results:
            #     print("albedo_path", albedo_path, "threshold", res[0], "f1", calc_metrics(res[1]["tp_num"], res[1]["fp_num"], res[1]["fn_num"]))
            #     pass
            
            albedo_rescaled_edges = result["albedo_edges"]
            img_resized_rescaled_edges = result["img_edges"]
            tp_num, fp_num, fn_num, debug = result["tp_num"], result["fp_num"], result["fn_num"], result["debug"]
            albedo_edges_vis[region_mask] = albedo_rescaled_edges[region_mask]
            img_edges_vis[region_mask] = img_resized_rescaled_edges[region_mask]
            print("tp_num", tp_num, "fp_num", fp_num, "fn_num", fn_num)
            tp_sum += tp_num
            fp_sum += fp_num
            fn_sum += fn_num
            tp_vis[region_mask] = np.logical_or(tp_vis[region_mask], debug["tp_img"][region_mask])
            fp_vis[region_mask] = np.logical_or(fp_vis[region_mask], debug["fp_img"][region_mask])
            fn_vis[region_mask] = np.logical_or(fn_vis[region_mask], debug["fn_img"][region_mask])
            #count_edges(albedo_rescaled)
            pass
        
        print("final albedo_path", albedo_path, "tp", tp_sum, "fp", fp_sum, "fn", fn_sum)        
        recall, precision, f1= calc_metrics(tp_sum, fp_sum, fn_sum)
        
        albedo_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_edges.png")
        albedo_gray_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_gray.png")
        img_resized_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_resized.png")
        img_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_edges.png")
        smooth_label_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_smooth_label.png")
        smooth_label_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_smooth_label.png")
        tp_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_tp.png")
        fp_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_fp.png")
        fn_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_fn.png")
        
        imageio.imwrite(albedo_gray_path, albedo_vis)
        imageio.imwrite(albedo_edges_path, albedo_edges_vis)
        imageio.imwrite(img_resized_path, img_vis)
        imageio.imwrite(img_edges_path, img_edges_vis)
        imageio.imwrite(smooth_label_path, smooth_label_resized.astype(np.uint8)*255)
        imageio.imwrite(tp_path, tp_vis*255)
        imageio.imwrite(fp_path, tp_vis*255)
        imageio.imwrite(fn_path, tp_vis*255)
        
        return recall, precision, f1
        pass
    pass
def shift_img(img, shift_x, shift_y):
    shape_y, shape_x = img.shape[:2]
    
    pad_y_before = max(-shift_y, 0)
    pad_y_after = max(shift_y, 0)

    pad_x_before = max(-shift_x, 0)
    pad_x_after = max(shift_x, 0)
    
    start_y, start_x = max(shift_y, 0), max(shift_x, 0)
    end_y, end_x = shape_y + max(shift_y, 0), shape_x + max(shift_x, 0)
    
    padded_img = np.pad(img, ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)))
    new_img = padded_img[start_y:end_y, start_x:end_x]
    assert (np.array(new_img.shape) == np.array(img.shape)).all(), (shift_x, shift_y, (pad_y_before, pad_y_after, pad_x_before, pad_x_after), new_img.shape, img.shape, padded_img.shape, (start_y, end_y, start_x, end_x))
    return new_img
    pass
class TextureEdgeEvaluatorV4:
    def __init__(self):
        pass
    
    def evaluate(self, img_path, label_path, albedo_path, albedo_srgb):
        print(img_path)
        img_pil = PIL.Image.open(img_path)
        img = np.array(img_pil)
        label = np.load(label_path)
        smooth_label = label == 2

        img_edges = cv2.Canny(img, 45, 45)
        img_edges_vis = np.tile(np.zeros_like(img_edges)[:, :, np.newaxis], [1, 1, 3])
        img_edges_vis[smooth_label] = np.array([255,0,0])
        img_edges_vis[img_edges == 255] = 255
        imageio.imwrite(os.path.join("vis_edges", os.path.basename(img_path)), img_edges_vis)

        
        albedo_pil = PIL.Image.open(albedo_path)
        albedo_tmp = np.array(albedo_pil) / 255
        if not albedo_srgb:
            albedo_tmp = albedolib.lin2srgb(albedo_tmp)
            pass
        if len(albedo_tmp.shape) == 2:
            albedo_tmp = np.tile(albedo_tmp[:, :, np.newaxis], [1,1,3])
            pass
        albedo_pil = PIL.Image.fromarray((albedo_tmp*255).round().astype(np.uint8))
        albedo = np.array(albedo_pil)
        
        albedo_resized = np.array(albedo_pil.resize(size=(img.shape[1], img.shape[0]), resample=PIL.Image.Resampling.LANCZOS))
        albedo_resized_linear = albedolib.srgb_to_linsrgb(albedo_resized / 255) 
        albedo_resized_linear_gray = albedo_resized_linear.mean(axis=-1)

        img_pil = PIL.Image.fromarray(img)
        img = np.array(img_pil)
        img_resized = img #np.array(img_pil.resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.LANCZOS))
        img_resized_linear = albedolib.srgb_to_linsrgb(img_resized / 255) 
        img_resized_linear_gray = img_resized_linear.mean(axis=-1)
        img_resized_gray = (albedolib.lin2srgb(img_resized_linear_gray) * 255).round().astype(np.uint8)
        smooth_label_resized = np.array(PIL.Image.fromarray(smooth_label).resize(size=(img.shape[1], img.shape[0]), resample=PIL.Image.Resampling.NEAREST)) #np.array(PIL.Image.fromarray(smooth_label).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.NEAREST))
        
        labels = skimage.measure.label(smooth_label_resized)
        #plt.imshow(labels)
        #plt.show()
        #exit()
        indices = np.unique(labels)

        tp_sum = 0
        fp_sum = 0
        fn_sum = 0

        # img_edges_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        # albedo_edges_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        # img_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        # albedo_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)

        # tp_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        # fp_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)
        # fn_vis = np.zeros((albedo.shape[0], albedo.shape[1]), dtype=np.uint8)

        img_edges_vis = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        albedo_edges_vis = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        img_vis = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        albedo_vis = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        tp_vis = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        fp_vis = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        fn_vis = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        def calc_metrics(tp_sum, fp_sum, fn_sum):
            recall = tp_sum / (tp_sum + fn_sum) if tp_sum+fn_sum != 0 else 0    
            precision = (tp_sum / (tp_sum + fp_sum)) if (tp_sum+fp_sum) != 0 else 0
            f1 = 2* (precision * recall) / (precision + recall) if (precision+recall) != 0 else 0
            return recall, precision, f1
        
        img_has_valid_patch = False
        for idx in indices[1:]:
            region_mask = labels == idx
            img_resized_linear_region_avg = img_resized_linear_gray[region_mask].mean()
            # img_resized_linear_rescaled = (img_resized_linear_gray / img_resized_linear_region_avg * 0.21404114)
            # img_resized_rescaled = (albedolib.lin2srgb(img_resized_linear_rescaled)*255).round().astype(np.uint8)
            # img_resized_rescaled_edges = cv2.Canny(img_resized_rescaled, 45, 45)
            img_resized_edges = cv2.Canny(img_resized_gray, 45, 45)
            
            print("albedopath", albedo_path, img_resized_edges[region_mask].mean() / 255, img_resized_edges[region_mask].sum() / 255)
            if  img_resized_edges[region_mask].sum() / 255 < 20:
                continue
            print(albedo_path, albedo_resized_linear_gray.shape, region_mask.shape, img_resized_linear_region_avg.shape)
            albedo_resized_linear_region_avg = albedo_resized_linear_gray[region_mask].mean()
            print(albedo_path, albedo_resized_linear_gray.shape, albedo_resized_linear_region_avg.shape, img_resized_linear_region_avg.shape)    
            albedo_resized_linear_rescaled = (albedo_resized_linear_gray / albedo_resized_linear_region_avg * img_resized_linear_region_avg )
            albedo_resized_rescaled = (albedolib.lin2srgb(albedo_resized_linear_rescaled) * 255).round().astype(np.uint8)
            
            albedo_vis[region_mask] = albedo_resized_rescaled[region_mask]
            img_vis[region_mask] = img_resized_gray[region_mask]
            
            img_has_valid_patch = True
            def sweep(threshold, shift_y, shift_x):
                # albedo_resized_rescaled_masked = albedo_resized_rescaled.copy()
                # region_mask_dil = skimage.morphology.binary_dilation(region_mask)
                # albedo_resized_rescaled_masked[np.logical_not(region_mask_dil)] = 0
                albedo_resized_rescaled_shifted = shift_img(albedo_resized_rescaled, shift_x, shift_y)
                
                #print(shift_y, shift_x)
                albedo_resized_rescaled_shifted_edges = cv2.Canny(albedo_resized_rescaled_shifted, threshold, threshold)    
                tp_num, fp_num, fn_num, debug = count_edges(albedo_resized_rescaled_shifted_edges, img_resized_edges, region_mask, debug=True)
                return {"albedo_edges": albedo_resized_rescaled_shifted_edges,
                        "img_edges": img_resized_edges,
                        "tp_num": tp_num,
                        "fp_num": fp_num,
                        "fn_num": fn_num,
                        "debug":debug,
                        }
            
            results = [(threshold, sweep(threshold, shift_y, shift_x), shift_y, shift_x) for threshold in range(30, 255, 10) for shift_y in range(-4, 5, 1) for shift_x in range(-4, 5, 1)]
            #results = [(threshold, sweep(threshold, shift_y, shift_x)) for threshold in range(30, 255, 5) for shift_y in range(-0, 1, 1) for shift_x in range(-0, 1, 1)]
            def res_cmp(threshold_result):
                tp, fp, fn = threshold_result[1]["tp_num"], threshold_result[1]["fp_num"], threshold_result[1]["fn_num"]
                recall,precision,f1=calc_metrics(tp, fp, fn)
                return (f1, recall, precision, tp, -fp, -fn)
            
            threshold, result, shift_y, shift_x = max(results, key=lambda threshold_result: res_cmp(threshold_result))
            print("albedo_path", albedo_path, "threshold", threshold, "shift", (shift_x, shift_y), "f1:", res_cmp((threshold, result)))
            # for res in results:
            #     print("albedo_path", albedo_path, "threshold", res[0], "f1", calc_metrics(res[1]["tp_num"], res[1]["fp_num"], res[1]["fn_num"]))
            #     pass
            #results = [(threshold_d, sweep(threshold_d, shift_y_d, shift_x_d), shift_y_d, shift_x_d) for threshold_d in range(max(threshold-10, 30), min(threshold+10, 255), 1) for shift_x_d in range(max(shift_x-2, -4), min(shift_x+2, 4)+1, 1) for shift_y_d in range(max(shift_y-2, -4), min(shift_y+2, 4)+1, 1)]
            #results = [(threshold_d, sweep(threshold_d, shift_y, shift_x), shift_y, shift_x) for threshold_d in range(max(threshold-4, 30), min(threshold+4, 255), 1)]
            results = [(threshold_d, sweep(threshold_d, shift_y, shift_x), shift_y, shift_x) for threshold_d in range(30, 255, 1)]
            threshold, result, shift_y, shift_x = max(results, key=lambda threshold_result: res_cmp(threshold_result))
            print("albedo_path", albedo_path, "threshold", threshold, "shift", (shift_x, shift_y), "f1", res_cmp((threshold, result)))
            
            albedo_resized_rescaled_edges = result["albedo_edges"]
            img_resized_edges = result["img_edges"]
            tp_num, fp_num, fn_num, debug = result["tp_num"], result["fp_num"], result["fn_num"], result["debug"]
            albedo_edges_vis[region_mask] = albedo_resized_rescaled_edges[region_mask]
            img_edges_vis[region_mask] = img_resized_edges[region_mask]
            print("tp_num", tp_num, "fp_num", fp_num, "fn_num", fn_num)
            tp_sum += tp_num
            fp_sum += fp_num
            fn_sum += fn_num

            tp_vis[region_mask] = np.logical_or(tp_vis[region_mask], debug["tp_img"][region_mask])
            fp_vis[region_mask] = np.logical_or(fp_vis[region_mask], debug["fp_img"][region_mask])
            fn_vis[region_mask] = np.logical_or(fn_vis[region_mask], debug["fn_img"][region_mask])
            #count_edges(albedo_rescaled)
            pass
        
        #print("final albedo_path", albedo_path, "tp", tp_sum, "fp", fp_sum, "fn", fn_sum)        
        if not img_has_valid_patch:
            return None
        
        recall, precision, f1= calc_metrics(tp_sum, fp_sum, fn_sum)
        albedo_resized_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_albedo_resized.png")
        albedo_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_edges.png")
        albedo_gray_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_gray.png")
        img_vis_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_vis.png")
        img_resized_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_resized.png")
        img_edges_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_img_edges.png")
        smooth_label_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_smooth_label.png")
        smooth_label_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_smooth_label.png")
        tp_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_tp.png")
        fp_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_fp.png")
        fn_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_fn.png")

        imageio.imwrite(albedo_resized_path, albedo_resized)
        imageio.imwrite(albedo_gray_path, albedo_vis)
        imageio.imwrite(albedo_edges_path, albedo_edges_vis)
        imageio.imwrite(img_resized_path, img_resized)
        imageio.imwrite(img_vis_path, img_vis)
        imageio.imwrite(img_edges_path, img_edges_vis)
        imageio.imwrite(smooth_label_path, smooth_label_resized.astype(np.uint8)*255)
        imageio.imwrite(tp_path, tp_vis*255)
        imageio.imwrite(fp_path, fp_vis*255)
        imageio.imwrite(fn_path, fn_vis*255)
        
        whole_gray_path = os.path.join(os.path.dirname(albedo_path), os.path.splitext(os.path.basename(albedo_path))[0]+"_whole_gray.png")
        albedo_resized_linear_gray_vis = (albedolib.lin2srgb(albedo_resized_linear_gray) * 255).round().astype(np.uint8)
        imageio.imwrite(whole_gray_path, albedo_resized_linear_gray_vis)
        return recall, precision, f1
        pass
    pass

class TextureEvaluatorLPIPS:
    def __init__(self, use_gpu=False):
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.scale_factor = 2
        pass
    def precompute(self, img_path, label_path):
        img_pil = PIL.Image.open(img_path)
        img = np.array(img_pil)
        label = np.load(label_path)
        smooth_label = label == 2
        img_pil = PIL.Image.fromarray(img)
        img = np.array(img_pil)
        img_resized_pil = img_pil.resize(size=(img.shape[1]*self.scale_factor, img.shape[0]*self.scale_factor), resample=PIL.Image.Resampling.LANCZOS)
        img_resized = np.array(img_resized_pil)
        
        smooth_label_resized = np.array(PIL.Image.fromarray(smooth_label).resize(size=(img.shape[1]*self.scale_factor, img.shape[0]*self.scale_factor), resample=PIL.Image.Resampling.NEAREST)) #np.array(PIL.Image.fromarray(smooth_label).resize(size=(albedo.shape[1], albedo.shape[0]), resample=PIL.Image.Resampling.NEAREST))
        
        labels = skimage.measure.label(smooth_label_resized)
        #plt.imshow(labels)
        #plt.show()
        #exit()
        indices = np.unique(labels)

        def get_lirarr(region_mask):
            region_mask_cv = region_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = contours[0][:, 0, :]
            lirarr = lir.lir(region_mask, contour) # format [x_left, y_upper, width, height]
            return lirarr
        lirarrs = []
        for idx in indices[1:]:
            region_mask = labels == idx            
            degrees = np.arange(-45, 45, 15)
            lirarr = get_lirarr(region_mask)
            print(lirarr)
            if lirarr[2] > 30 and lirarr[3] > 30:
                lirarrs.append(lirarr)
                pass
            pass
        return lirarrs, img_resized
    
    def evaluate_from_precompute(self, lirarrs, img_resized, albedo_path, albedo_srgb):
        print(albedo_path)
        
        albedo_pil = PIL.Image.open(albedo_path)
        albedo_tmp = np.array(albedo_pil) / 255
        if not albedo_srgb:
            albedo_tmp = albedolib.lin2srgb(albedo_tmp)
            pass
        if len(albedo_tmp.shape) == 2:
            albedo_tmp = np.tile(albedo_tmp[:, :, np.newaxis], [1,1,3])
            pass
        albedo_pil = PIL.Image.fromarray((albedo_tmp*255).round().astype(np.uint8))
        
        albedo_resized = np.array(albedo_pil.resize(size=(img_resized.shape[1], img_resized.shape[0]), resample=PIL.Image.Resampling.LANCZOS))
        
        scores = []
        counter = 0
        img_has_valid_patch = False
        for idx, lirarr in enumerate(lirarrs):
        
            img_crop = img_resized[lirarr[1]:(lirarr[1]+lirarr[3]), lirarr[0]:(lirarr[0]+lirarr[2])]
            albedo_crop = albedo_resized[lirarr[1]:(lirarr[1]+lirarr[3]), lirarr[0]:(lirarr[0]+lirarr[2])]
            
            img_crop_linear = albedolib.srgb_to_linsrgb(img_crop / 255)
            albedo_crop_linear = albedolib.srgb_to_linsrgb(albedo_crop / 255)
            
            ch1_shade = np.mean(img_crop_linear[:,:,0])/np.mean(albedo_crop_linear[:,:,0])
            ch2_shade = np.mean(img_crop_linear[:,:,1])/np.mean(albedo_crop_linear[:,:,1])
            ch3_shade = np.mean(img_crop_linear[:,:,2])/np.mean(albedo_crop_linear[:,:,2])
            print(ch1_shade, ch2_shade, ch3_shade)
            albedo_crop_linear[:,:,0] = albedo_crop_linear[:,:,0] * ch1_shade
            albedo_crop_linear[:,:,1] = albedo_crop_linear[:,:,1] * ch2_shade
            albedo_crop_linear[:,:,2] = albedo_crop_linear[:,:,2] * ch3_shade
            albedo_crop_srgb = (albedolib.lin2srgb(albedo_crop_linear)*255).round().astype(np.uint8)
            
            if lirarr[2] > 30 and lirarr[3] > 30:
                
                print("acc")
                img_has_valid_patch = True
                d = self.loss_fn.forward(lpips.im2tensor(img_crop).to(self.device) , lpips.im2tensor(albedo_crop_srgb).to(self.device)).item()
                scores.append(d)
                img_resized_write_path = os.path.splitext(albedo_path)[0] + f"_idx_{counter}_img_resized.png"
                imageio.imwrite(img_resized_write_path, img_resized, compress_level=0)

                albedo_resized_write_path = os.path.splitext(albedo_path)[0] + f"_idx_{counter}_resized.png"
                imageio.imwrite(albedo_resized_write_path, albedo_resized, compress_level=0)

                img_crop_write_path = os.path.splitext(albedo_path)[0] + f"_idx_{counter}_img_crop.png"
                imageio.imwrite(img_crop_write_path, img_crop, compress_level=0)

                albedo_crop_write_path = os.path.splitext(albedo_path)[0] + f"_idx_{counter}_albedo_crop.png"
                imageio.imwrite(albedo_crop_write_path, albedo_crop, compress_level=0)

                albedo_crop_reshade_write_path = os.path.splitext(albedo_path)[0] + f"_idx_{counter}_albedo_crop_reshade.png"
                imageio.imwrite(albedo_crop_reshade_write_path, albedo_crop_srgb, compress_level=0)
                counter+=1
                pass
            
            img_resized_write_path = os.path.splitext(albedo_path)[0] + f"_idx_all_{idx}_img_resized.png"
            imageio.imwrite(img_resized_write_path, img_resized, compress_level=0)

            albedo_resized_write_path = os.path.splitext(albedo_path)[0] + f"_idx_all_{idx}_resized.png"
            imageio.imwrite(albedo_resized_write_path, albedo_resized, compress_level=0)

            img_crop_write_path = os.path.splitext(albedo_path)[0] + f"_idx_all_{idx}_img_crop.png"
            imageio.imwrite(img_crop_write_path, img_crop, compress_level=0)

            albedo_crop_write_path = os.path.splitext(albedo_path)[0] + f"_idx_all_{idx}_albedo_crop.png"
            imageio.imwrite(albedo_crop_write_path, albedo_crop, compress_level=0)

            albedo_crop_reshade_write_path = os.path.splitext(albedo_path)[0] + f"_idx_all_{idx}_albedo_crop_reshade.png"
            imageio.imwrite(albedo_crop_reshade_write_path, albedo_crop_srgb, compress_level=0)
            pass
        if img_has_valid_patch:
            return np.mean(scores), scores
        else:
            return None
        pass
    def evaluate(self, img_path, smooth_label_path, albedo_path, is_srgb):
        lirarrs, img_resized = self.precompute(img_path, smooth_label_path)
        res = self.evaluate_from_precompute(lirarrs, img_resized, albedo_path, is_srgb)
        return res
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["saw", "maw"], type=str)
    parser.add_argument("--meta", type=str)
    parser.add_argument("--imgs_dir", type=str, default=None)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    evaluator = TextureEvaluatorLPIPS(use_gpu=args.use_gpu)
    
    if args.dataset == "saw":
        assert args.meta is not None
        pass
    
    if args.dataset == "saw":
        #mgr = utils.PathFinderSawMgr("/data/research/datasets/saw/picked.txt")
        ids=np.load("/data/research/datasets/saw/saw_splits/test_ids.npy")
        #ids = ids[:10]
        ids = [id for id in ids if id < 118520]
        ids.sort()
        #ids = [2894]
        #ids=ids[:5]
        mgr = utils.PathFinderSawMgr([str(id) for id in ids])
        pass
    elif args.dataset == "maw":
        mgr = utils.PathFinderMgr(args.meta, imgs_dir=args.imgs_dir)
        pass
    else:
        raise NotImplementedError(args.dataset)
    
    names = ["ravi", "ravi_iiw", "ravi_iiw_bs", "cgintrinsics", "cgintrinsics_filtered", "bigtime", "soumyadip", "usi3d", "revisit", "revisit_prime", "bell2014", "NIID", "nestmeyer", "nestmeyer_filtered"]
    meta = {}
    def run(row):
        if not os.path.exists(row.get_smooth_label_path()):
                print("missing", row.get_smooth_label_path())
                return None
            
        lirarrs, img_resized = evaluator.precompute(row.get_img_path(), row.get_smooth_label_path())
        results = {}
        for name in names:
            result = row.get_albedo(name)
            if name == "baseline":
                albedo_path, color_space = result, "srgb"
                pass
            else:
                albedo_path, color_space = result
                pass
            
            if not os.path.exists(albedo_path) :
                raise RuntimeError(albedo_path)
                return None
            is_srgb = None
            if color_space == "srgb":
                is_srgb=True
                pass
            elif color_space == "linear":
                is_srgb=False
                pass
            else:
                raise NotImplementedError(color_space)
            res = evaluator.evaluate_from_precompute(lirarrs, img_resized, albedo_path, is_srgb)
            if res is None:
                return None
            lpips, all_lpips = res
            results[name] = {
                "lpips": lpips,
                "all_lpips": all_lpips, 
                "albedo": albedo_path,
                "image": row.get_img_path()
            }
            pass
        lengths = [len(v["all_lpips"]) for k,v in results.items()]
        print(lengths)
        assert len(set(lengths)) == 1
        return results

    meta = list(map(run, [ex for ex in list(mgr.iter_examples())]))
    meta_filtered = [row for row in meta if row is not None]
    outputfile = "output_tex_lpips.csv"
    if args.dataset == "maw":
        outputfile = "output_tex_lpips_maw_{}.csv".format(os.path.basename(args.meta))
        pass
        
    with open(outputfile, "wt") as f:
        writer = csv.writer(f)
        for row in meta_filtered:
            row_total = []
            for type in ["lpips"]:
                row_total.extend([row[name][type] for name in names])
                pass
            writer.writerow(row_total)
            pass
        
        pass
    

    length = sum([len(res[names[0]]["all_lpips"]) for res in meta_filtered])
    imgs = [res[names[0]]["image"] for res in meta_filtered for i in range(len(res[names[0]]["all_lpips"]))]
    for i, start in enumerate(range(0, length, 50)):
        end = start + 50
        cols = [
            Col('id1',  'ID'),                                               # make a column of 1-based indices
            Col('img',  'input', imgs, subset=(start, end) ),             # specify image content for column 2
        ]
        for name in names:
            
            albedo = [res[name]["albedo"] for res in meta_filtered for i in range(len(res[name]["all_lpips"]))]
            cols.append(
                Col('img',  '{}_albedo'.format(name), albedo, subset=(start, end) )
            )
            img_resized = [os.path.splitext(res[name]["albedo"])[0] + f"_idx_{i}_img_resized.png" for res in meta_filtered for i in range(len(res[name]["all_lpips"]))]
            cols.append(
                Col('img',  '{}_img_resized'.format(name), img_resized, subset=(start, end) )
            )
            
            albedo_resized = [os.path.splitext(res[name]["albedo"])[0] + f"_idx_{i}_resized.png" for res in meta_filtered for i in range(len(res[name]["all_lpips"]))]
            cols.append(
                Col('img',  '{}_albedo_resized'.format(name), albedo_resized, subset=(start, end) )
            )

            img_crop = [os.path.splitext(res[name]["albedo"])[0] + f"_idx_{i}_img_crop.png" for res in meta_filtered for i in range(len(res[name]["all_lpips"]))]
            cols.append(
                Col('img',  '{}_img_crop'.format(name), img_crop, subset=(start, end) )
            )

            albedo_crop = [os.path.splitext(res[name]["albedo"])[0] + f"_idx_{i}_albedo_crop.png" for res in meta_filtered for i in range(len(res[name]["all_lpips"]))]
            cols.append(
                Col('img',  '{}_albedo_crop'.format(name), albedo_crop, subset=(start, end) )
            )

            albedo_crop_reshade = [os.path.splitext(res[name]["albedo"])[0] + f"_idx_{i}_albedo_crop_reshade.png" for res in meta_filtered for i in range(len(res[name]["all_lpips"]))]

            cols.append(
                Col('img',  '{}_albedo_crop_reshade'.format(name), albedo_crop_reshade, subset=(start, end) )
            )

            lpips = [lpips for res in meta_filtered for lpips in res[name]["all_lpips"]]
            cols.append(
                Col('text',  '{}_lpips'.format(name), lpips, subset=(start, end) )
            )
            pass
        imagetable(cols, out_file="index_reduced_lpips_texture_{}.html".format(i), imsize=[320, 240])


if __name__ == "__main__":
    main()


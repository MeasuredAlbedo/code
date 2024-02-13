import numpy as np
import argparse
import os
import imageio
import albedolib
import colour
import PIL
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import pathlib
import math
import compute_shading

def scale_for_si_mse(x, y):
    theta = 1/np.maximum(np.sum(x*x), 1e-6)*np.sum(x*y)
    return theta

def scale_for_weighted_si_mse(w, x, y):
    theta = 1/np.maximum(np.sum(w*x*x), 1e-6)*np.sum(w*x*y)
    return theta

class AlbedoEvaluator:
    def __init__(self, loss, metric):
        self.loss = loss
        self.metric = metric
        pass
    def get_scaled_albedo_path(self, p):
        if self.loss == "si":
            return os.path.splitext(p)[0]+"_scaled.png"
            pass
        elif self.loss == "per_si":
            return os.path.splitext(p)[0]+"_per_scaled.png"
            pass
        elif self.loss == "wb_per_si":
            return os.path.splitext(p)[0]+"_wb_per_scaled.png"
            pass
        elif self.loss == "reg":
            return os.path.splitext(p)[0]+"_reg.png"
            pass
        else:
            raise NotImplementedError(self.loss)
        
        pass
    def get_grayscale_albedo_path(self, p):
        return os.path.splitext(p)[0]+"_gray.png"
        pass

    def get_grayscale_scaled_albedo_path(self, p):
        return os.path.splitext(p)[0]+"_gray_scaled.png"
        pass
    
    def evaluate(self, input_path_colors, input_path_masks, pred_albedo, color_space):
        colors = np.array(np.load(input_path_colors))
        if pred_albedo != "baseline_c":
            albedo_pred = imageio.imread(pred_albedo)
            albedo_pred = PIL.Image.fromarray(albedo_pred)
            albedo_pred = albedo_pred.resize((320, 240), resample=PIL.Image.BILINEAR)
            albedo_pred = np.array(albedo_pred)
            if color_space == "linear":
                albedo_pred = (albedolib.lin2srgb(albedo_pred / 255) * 255).round() 
                pass
            elif color_space == "ravi":
                raise NotImplementedError(color_space)
            elif color_space == "srgb":
                pass # do nothing
            else:
                raise NotImplementedError(color_space)
            if len(albedo_pred.shape) == 2:
                albedo_pred = np.tile(albedo_pred[:, :, np.newaxis], (1, 1, 3))
                pass
        else:
            albedo_pred = np.full((240, 320, 3), 127, dtype=np.uint8)
            pass
        
        mask = imageio.imread(input_path_masks)
        mask = PIL.Image.fromarray(mask)
        mask = mask.resize((320, 240), resample=PIL.Image.NEAREST)
        mask = np.array(mask)
        
        indices = np.unique(mask)

        medians = []
        srgb_gts = []
        weights = []
        scaled_albedo = albedolib.srgb_to_linsrgb(albedo_pred.copy() / 255)
        for i, idx in enumerate(indices[1:]):        
            median = albedolib._find_robust_median(albedolib.srgb_to_linsrgb(albedo_pred[mask==idx]/255))
            medians.append(median)
            weights.append((mask==idx).sum())
            srgb_gt = albedolib.adobelin2srgb(colors[i])
            srgb_gts.append(albedolib.srgb_to_linsrgb(srgb_gt))
            pass
        weights = np.array(weights)
        if self.loss == "si":
            if self.metric == "mean_srgb":
                theta = scale_for_weighted_si_mse(weights, albedolib.lin2srgb(np.array(medians).mean(axis=-1)), albedolib.lin2srgb(np.array(srgb_gts).mean(axis=-1)))
                for i, (median, idx) in enumerate(zip(medians, indices[1:])):
                    medians[i] = theta * albedolib.lin2srgb(median.mean(axis=-1))
                pass
            elif self.metric == "no_mean":
                theta = scale_for_weighted_si_mse(weights[:, np.newaxis].repeat(3, axis=-1), np.array(medians), np.array(srgb_gts))
                for i, (median, idx) in enumerate(zip(medians, indices[1:])):
                    medians[i] = theta * median
                    scaled_albedo[mask==idx] = medians[i]
                    pass

            else:
                theta = scale_for_weighted_si_mse(weights, np.array(medians).mean(axis=-1), np.array(srgb_gts).mean(axis=-1))
                for i, (median, idx) in enumerate(zip(medians, indices[1:])):
                    medians[i] = theta * median
                    scaled_albedo[mask==idx] = medians[i]
                    pass
                pass
            
        elif self.loss == "per_si":
            for i, (median, idx) in enumerate(zip(medians, indices[1:])):
                theta = scale_for_si_mse(np.array([median]).mean(), np.array([srgb_gts[i]]).mean())
                medians[i] = theta * median
                scaled_albedo[mask==idx] = medians[i]
                pass
            pass
        elif self.loss == "wb_per_si":
            medians_np = np.array(medians) #Nx3
            srgb_gts_np = np.array(srgb_gts)
            scale = [scale_for_si_mse(medians_np[:, i], srgb_gts_np[:, i]) for i in range(3)]
            for i in range(3):
                assert len(medians) == len(srgb_gts)
                for j in range(len(medians)):
                    medians[j][i] = scale[i] * medians[j][i]
                    pass
                pass                    
            for i, (median, idx) in enumerate(zip(medians, indices[1:])):
                theta = scale_for_si_mse(np.array([median]), np.array([srgb_gts[i]]))
                medians[i] = theta * median
                scaled_albedo[mask==idx] = medians[i]
                pass
            pass
        elif self.loss == "reg":
            for i, (median, idx) in enumerate(zip(medians, indices[1:])):
                scaled_albedo[mask==idx] = medians[i]
                pass
        else:
            raise NotImplementedError(self.loss)
        if self.metric != "mean_srgb":
            imageio.imwrite(self.get_scaled_albedo_path(pred_albedo), np.round(albedolib.lin2srgb(np.clip(scaled_albedo , 0.0, 1.0))*255).astype(np.uint8), compress_level=1)
            imageio.imwrite(self.get_grayscale_albedo_path(pred_albedo), (albedolib.lin2srgb(np.mean(albedolib.srgb_to_linsrgb(albedo_pred / 255), axis=-1)) * 255).round().astype(np.uint8))
            if self.loss == "si":
                imageio.imwrite(self.get_grayscale_scaled_albedo_path(pred_albedo), (albedolib.lin2srgb(np.mean(theta * albedolib.srgb_to_linsrgb(albedo_pred / 255), axis=-1)) * 255).round().astype(np.uint8), compress_level=1)
                pass
            pass
        
        
        delta_es = []
        for i, (median, gt) in enumerate(zip(medians, srgb_gts)):
            if self.metric == "deltae":
                lab_pred = albedolib.srgb_to_lab(albedolib.lin2srgb(median))
                lab_gt = albedolib.srgb_to_lab(albedolib.lin2srgb(gt))
                deltae=colour.delta_E(lab_gt, lab_pred)
                delta_es.append(deltae)
                pass
            elif self.metric[:3] == "lab":
                lab_pred = albedolib.srgb_to_lab(albedolib.lin2srgb(median))
                lab_gt = albedolib.srgb_to_lab(albedolib.lin2srgb(gt))

                if self.metric == "lab_l":
                    delta = np.abs(lab_pred[0] - lab_gt[0])
                elif self.metric == "lab_a":
                    delta = np.abs(lab_pred[1] - lab_gt[1])
                    pass
                elif self.metric == "lab_b":
                    delta = np.abs(lab_pred[2] - lab_gt[2])
                    pass
                else:
                    raise NotImplementedError("unreacheable")
                delta_es.append(delta)
                pass
            elif self.metric == "no_mean":
                assert len(median) == 3, median
                assert len(gt) == 3, gt

                mean_pred = median
                mean_gt = gt

                
                delta_es.append(np.square(mean_gt - mean_pred).mean())
            elif self.metric == "diff":
                assert len(median) == 3, median
                assert len(gt) == 3, gt

                mean_pred = median.mean()
                mean_gt = gt.mean()

                
                delta_es.append((mean_gt - mean_pred))
                
            elif self.metric == "mean":
                assert len(median) == 3, median
                assert len(gt) == 3, gt
                # mean_pred = np.clip(median.mean(), 0, 1)
                # mean_gt = np.clip(gt.mean(), 0, 1)
                mean_pred = median.mean()
                mean_gt = gt.mean()

                
                delta_es.append(np.square(mean_gt - mean_pred))
            elif self.metric == "mean_srgb":
                assert np.array(median).size == 1, median
                assert np.array(gt).size == 3, gt
                mean_pred = median
                mean_gt = albedolib.lin2srgb(gt.mean())
                
                delta_es.append(np.square(mean_gt - mean_pred))
                pass
            elif self.metric == "accuracy":
                mean_pred = median.mean()
                mean_gt = gt.mean()
                
                if not (((mean_pred / mean_gt) > 1.2) or ((mean_gt / mean_pred) > 1.2)):
                    delta_es.append(1.0)
                    pass
                else:
                    delta_es.append(0.0)
                pass
            else:
                lab_pred = albedolib.srgb_to_hsv(albedolib.lin2srgb(median))
                lab_gt = albedolib.srgb_to_hsv(albedolib.lin2srgb(gt))
                if self.metric == "hue":
                    #delta_hsv = np.abs(lab_pred[0] - lab_gt[0])
                    max_num = max(lab_pred[0], lab_gt[0])
                    min_num = min(lab_pred[0], lab_gt[0])
                    delta1 = np.abs(max_num-min_num)
                    delta2 = np.abs((max_num-360)-min_num)
                    delta_hsv = delta1 if delta1 < delta2 else delta2
                    print("delta_hsv", delta_hsv)
                    pass
                elif self.metric == "saturation":
                    delta_hsv = np.abs(lab_pred[1] - lab_gt[1])
                    pass
                elif self.metric == "value":
                    delta_hsv = np.abs(lab_pred[2] - lab_gt[2])
                    pass

                else:
                    raise NotImplementedError("unreacheable")
                
                delta_es.append(delta_hsv)
                #pass
            pass
        delta_es = np.array(delta_es)
        return delta_es, weights
    

    pass

def srgb_to_rgb(srgb):
    """ Convert an sRGB image to a linear RGB image """

    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret
def compute_whdr(reflectance, judgements, delta=0.10, return_results=False):
    """ Return the WHDR score for a reflectance image, evaluated against human
    judgements.  The return value is in the range 0.0 to 1.0, or None if there
    are no judgements for the image.  See section 3.5 of our paper for more
    details.

    :param reflectance: a numpy array containing the linear RGB
    reflectance image.

    :param judgements: a JSON object loaded from the Intrinsic Images in
    the Wild dataset.

    :param delta: the threshold where humans switch from saying "about the
    same" to "one point is darker."
    """

    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    weight_sum = 0.0
    results = []
    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[
            int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[
            int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker != alg_darker:
            error_sum += weight
            results.append(False)
            pass
        else:
            results.append(True)
        weight_sum += weight

    if weight_sum:
        if return_results:
            return error_sum / weight_sum, results
        else:
            return error_sum / weight_sum
    else:
        return None

def draw_whdr(image_intrinsic, judgement, results):
    albedo_srgb = image_intrinsic.copy()
    comparisons = judgement["intrinsic_comparisons"]
    intrinsic_points = judgement['intrinsic_points']
    id_to_points = {p['id']: p for p in intrinsic_points}
    for c, res in zip(comparisons, results):
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue
        rows,cols,_ = image_intrinsic.shape
        #print(rows, cols)
        pt1 = (int(point1['x']*cols), int(point1['y']*rows))
        pt2 = (int(point2['x']*cols), int(point2['y']*rows))
        
        pt1_col = (0, 0, 255)
        pt2_col = (255, 0, 0)

        if darker == '1': #p2 > p1
            pass
        elif darker == '2':
            pt1, pt2 = pt2, pt1
            pass
        else:
            pt1_col, pt2_col = (0, 255, 0), (0, 255, 0) 
            pass

        line = np.array(pt2) - np.array(pt1)
        pt1_shifted = tuple((np.array(pt1) + line*0.2).astype(int).tolist())
        pt2_shifted = tuple((np.array(pt1) + line*0.8).astype(int).tolist())
        center = tuple((np.array(pt1) + line*0.5).astype(int).tolist())
        
        if res:
            line_col = (0, 255, 0)
            pass
        else:
            line_col = (255, 0, 0)
            pass

        # cv2.arrowedLine(image_intrinsic, pt1, pt2, line_col, 1, tipLength=0.2)
        # cv2.arrowedLine(image_intrinsic, pt2, pt1, line_col, 1, tipLength=0.2)
        cv2.line(image_intrinsic, pt1, pt2, line_col, 1)
        cv2.circle(image_intrinsic, pt1_shifted, 1, pt1_col)
        cv2.circle(image_intrinsic, pt2_shifted, 1, pt2_col)
        
        image_intrinsic[pt1[1], pt1[0]] = albedo_srgb[pt1[1], pt1[0]]
        image_intrinsic[pt2[1], pt2[0]] = albedo_srgb[pt2[1], pt2[0]]
        pass


class WHDREvaluator(AlbedoEvaluator):
    def __init__(self):
        super().__init__("reg", "mean")
        pass
    def get_scaled_albedo_path(self, albedo):
        return albedo.replace(".png", "_label.png")
    def evaluate(self, input_judgement, pred_albedo, delta=0.10, srgb=True):
        with open(input_judgement) as f:
            judgement_obj = json.load(f)
            pass
        if pred_albedo == "baseline_c":
            albedo_pred = np.full((240, 320, 3), 127, dtype=np.uint8)
            pass
        else:
            albedo_pred = imageio.imread(pred_albedo)
            albedo_pred = PIL.Image.fromarray(albedo_pred)
            albedo_pred = np.array(albedo_pred)
            albedo_pred = albedo_pred / 255
            if srgb:
                albedo_pred = srgb_to_rgb(albedo_pred)
                pass
            if  len(albedo_pred.shape) == 2:
                albedo_pred = np.tile(albedo_pred[:, :, np.newaxis], [1, 1, 3])
                pass
            pass
        whdr_score, results = compute_whdr(albedo_pred, judgement_obj, delta=delta, return_results=True)
        albedo_srgb = (albedolib.lin2srgb(albedo_pred) * 255).round().astype(np.uint8)
        albedo_srgb = cv2.resize(albedo_srgb, (320, 240))
        draw_whdr(albedo_srgb, judgement_obj, results)
        if pred_albedo != "baseline_c":
            imageio.imwrite(pred_albedo.replace(".png", "_label.png"), albedo_srgb, compress_level=1)
        return np.array([whdr_score])
        pass
    pass

class ShadingEvaluator(AlbedoEvaluator):
    def __init__(self):
        super().__init__("reg", "mean")
        self.ver = "ver2" #ver1: blur then shading. ver2: shading then masked blur
        pass
    def evaluate(self, gt_shading_path, pred_shading_path, mask_path, specular_mask_path, input_img_path, color_space, sigma_factor=1): # Input img for check over exposure
        #WIDTH, HEIGHT = 320,240
        WIDTH = 320
        ORIG_SIGMA = 20
        
        input_img = PIL.Image.open(input_img_path)

        input_img_size = input_img.size # width, height
        new_width = WIDTH
        scale_factor = (new_width / input_img_size[0])
        new_height = int(round(input_img_size[1] * scale_factor))        
        new_sigma = 20 * scale_factor / sigma_factor
        input_img = input_img.resize((new_width, new_height))
        input_img = np.array(input_img)

        pred_shading = PIL.Image.open(pred_shading_path)        
        pred_shading = pred_shading.resize((new_width, new_height), resample=PIL.Image.BILINEAR)
        pred_shading = np.array(pred_shading) / 255
        if color_space == "linear":
            pass # do nothing
        elif color_space == "srgb":
            pred_shading = (albedolib.srgb2lin(pred_shading))
            pass
        else:
            raise NotImplementedError(color_space)
        
        if len(pred_shading.shape) == 2:
            pred_shading = np.tile(pred_shading[:, :, np.newaxis], (1,1,3))
            pass
        albedo_mask = PIL.Image.open(mask_path)
        albedo_mask = albedo_mask.resize((new_width, new_height), resample=PIL.Image.NEAREST)
        albedo_mask = np.array(albedo_mask)
        
        gt_shading = np.load(gt_shading_path, allow_pickle=True)["s"]
        gt_shading = cv2.resize(gt_shading, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
        gt_shading = compute_shading.naive_blur(gt_shading, albedo_mask, sigma=new_sigma)
                
        mask_albedo_def = albedo_mask != 0
                
        overexposure_mask = np.all(input_img < 250, axis=-1)
        
        if os.path.exists(specular_mask_path):
            specular_mask = PIL.Image.open(specular_mask_path)
            specular_mask = specular_mask.resize((new_width, new_height), resample=PIL.Image.NEAREST)
            specular_mask = np.array(specular_mask)
            specular_mask = specular_mask != 0
            pass
        else:
            specular_mask = np.zeros((new_height, new_width), dtype=np.bool_)
            pass
        specular_mask = ~specular_mask
        mask = np.logical_and(mask_albedo_def, overexposure_mask)
        mask = np.logical_and(mask, specular_mask)
        pred_shading_blur = compute_shading.naive_blur(pred_shading, albedo_mask, sigma=new_sigma)
        
        pred_values = pred_shading_blur[mask]
        gt_values = gt_shading[mask]
        
        theta = scale_for_si_mse(pred_values, gt_values)
        
        err = np.mean(((theta*pred_values) - gt_values)**2)
        
        #debug writes
        pred_shading_pathlib = pathlib.Path(pred_shading_path)
        pred_gt_max = max(pred_shading.max(), gt_shading.max())
                
        blur_shading_path = pred_shading_pathlib.parent / (pred_shading_pathlib.stem + "_blur_pred_shading.png")
        imageio.imwrite(blur_shading_path, albedolib.lin2srgb(pred_shading_blur/pred_gt_max))

        blur_shading_path = pred_shading_pathlib.parent / (pred_shading_pathlib.stem + "_blur_pred_shading_scaled.png")
        imageio.imwrite(blur_shading_path, albedolib.lin2srgb(theta*pred_shading_blur/pred_gt_max))
        
        gt_shading_path = pred_shading_pathlib.parent / (pred_shading_pathlib.stem + "_gt_shading.png")
        imageio.imwrite(gt_shading_path, albedolib.lin2srgb(gt_shading/pred_gt_max))

        mask_albedo_def_path = pred_shading_pathlib.parent / (pred_shading_pathlib.stem + "_mask_albedo_def.png")
        imageio.imwrite(mask_albedo_def_path, mask_albedo_def.astype(np.uint8)*255)

        overexposure_mask_path = pred_shading_pathlib.parent / (pred_shading_pathlib.stem + "_overexposure_mask.png")
        imageio.imwrite(overexposure_mask_path, overexposure_mask.astype(np.uint8)*255)

        specular_mask_path = pred_shading_pathlib.parent / (pred_shading_pathlib.stem + "_specular_mask.png")
        imageio.imwrite(specular_mask_path, specular_mask.astype(np.uint8)*255)

        mask_path = pred_shading_pathlib.parent / (pred_shading_pathlib.stem + "_mask.png")
        imageio.imwrite(mask_path, mask.astype(np.uint8)*255)
        return err
    pass


LOSS_CHOICES = ['reg', 'si', 'per_si', "wb_per_si"]
LOSS_DEFAULT = "si"

METRIC_CHOICES = ['deltae', 'hue', 'saturation', "value", "lab_l", "lab_a", "lab_b", "mean", "mean_srgb", "accuracy", "no_mean", "diff"]
METRIC_DEFAULT = "hue"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path_colors")
    parser.add_argument("input_path_masks")
    parser.add_argument("pred_albedo")
    parser.add_argument("--loss", type=str, default=LOSS_DEFAULT, choices=LOSS_CHOICES)
    
    parser.add_argument("--metric", type=str, default=METRIC_DEFAULT, choices=METRIC_CHOICES)
    
    args = parser.parse_args()
    
    evaluator = AlbedoEvaluator(loss=args.loss, metric=args.metric)
    metric = evaluator.evaluate(args.input_path_colors, args.input_path_masks, args.pred_albedo)
    print("log: ", metric)
    print("mean:", np.mean(metric, axis=0))

import subprocess
import csv
import argparse
import os
import utils
import collections
from html4vision import Col, imagetable
import numerical_albedo
#import texture_score
import numpy as np
import multiprocessing as mp
def run_row(row):
    mask = row.get_mask()
    img_path = row.get_img_path()
    gt_shading = row.get_gt_shading_path()
    gt_shading_vis = row.get_gt_shading_vis_path()
    specular_mask_path = row.get_specular_mask_path()
    meta_dict = {}
    
    meta_dict["img"] = img_path
    meta_dict["gt_shading"] = gt_shading_vis
    meta_dict["mask"] = mask

    for name in names:
        #ours
        result = row.get_shading(name)
        if name == "baseline":
            pred_shading, color_space = result, "srgb"
            pass
        else:
            pred_shading, color_space = result
            pass
        
        metric = evaluator.evaluate(gt_shading, pred_shading, mask, specular_mask_path, img_path, color_space)

        meta_dict[name] = {
            "pred_shading": pred_shading,
            "number": metric
        }
        pass
    return meta_dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="meta.csv")
    parser.add_argument("--imgs_dir", type=str, default=None)
    args = parser.parse_args()
    mgr = utils.PathFinderMgr(args.meta, imgs_dir=args.imgs_dir)
    evaluator = numerical_albedo.ShadingEvaluator()
    
    # meta_dict = collections.defaultdict(lambda: [])
    #names = ["baseline", "baseline_c", "ravi", "ravi_bs", "ravi_iiw", "ravi_iiw_bs", "cgintrinsics", "cgintrinsics_filtered", "bigtime","soumyadip", "usi3d", "revisit", "revisit_prime", "bell2014", "NIID", "nestmeyer", "nestmeyer_filtered"]#, "flatten"]#, "nonlocal"] #"l1trans"
    #names = ["cgintrinsics", "bigtime","usi3d",  "bell2014", "NIID"]#, "flatten"]#, "nonlocal"] #"l1trans"
    #names = ["NIID"]
    names = ["revisit", "nestmeyer", "soumyadip", "ravi_iiw_bs"]
    #names = ["ravi_iiw_bs"]
    with mp.Pool() as p:
        meta = p.map(run_row, list(mgr.iter_examples()))
        pass
    
    suffix = "{}".format(os.path.splitext(os.path.basename(args.meta))[0])
    #output_csv = "output_{}.csv".format(suffix) if args.output_file is None else args.output_file
    output_csv = "output_shading_{}.csv".format(suffix) 
    with open(output_csv, "wt") as f:
        writer = csv.writer(f)
        for row in meta:
            fn = row["img"]
            dn = os.path.basename(os.path.dirname(fn))
            bn = os.path.basename(fn)
            numbers = [row[name]["number"] for name in names]
            writer.writerow(["{}_{}".format(dn, bn)]+numbers)
            pass
        pass
    
    #print(our_numbers)
    #print(ravi_numbers)
    for i, start in enumerate(range(0, len(meta), 50)):
        print(start)
        end = start + 50
        cols = [
            Col('id1',  'ID'),                                               # make a column of 1-based indices
            Col('img',  'input', [row["img"] for row in meta], subset=(start, end)),             # specify image content for column 2
            Col('img',  'gt shading', [row["gt_shading"] for row in meta], subset=(start, end)),             # specify image content for column 2
            Col('img',  'mask', [row["mask"] for row in meta], subset=(start, end)),             # specify image content for column 2

            # Col('img',  'ours_albedo', meta_dict["ours"]["albedo"]),             # specify image content for column 2
            # Col('text', 'ours_log', meta_dict["ours"]["log"]),     # specify image content for column 3

            # Col('img',  'ravi_albedo', meta_dict["ravi"]["albedo"]),             # specify image content for column 2
            # Col('text', 'ravi_log', meta_dict["ravi"]["log"]),     # specify image content for column 3

            # Col('img',  'cgintrinsics_albedo', meta_dict["cgintrinsics"]["albedo"]),             # specify image content for column 2
            # Col('text', 'cgintrinsics_log', meta_dict["cgintrinsics"]["log"]),     # specify image content for column 3

            # Col('img',  'bigtime_albedo', meta_dict["bigtime"]["albedo"]),             # specify image content for column 2
            # Col('text', 'bigtime_log', meta_dict["bigtime"]["log"]),     # specify image content for column 3

            # Col('img',  'soumyadip_albedo', meta_dict["soumyadip"]["albedo"]),             # specify image content for column 2
            # Col('text', 'soumyadip_log', meta_dict["soumyadip"]["log"]),     # specify image content for column 3        
        ]

        for name in names:
            cols.append(Col('img',  '{}_shading'.format(name), [row[name]["pred_shading"] for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_blur_shading'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_blur_pred_shading.png" for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_gt_shading'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_gt_shading.png" for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_blur_shading_scaled'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_blur_pred_shading_scaled.png" for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_gt_shading'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_gt_shading.png" for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_mask_albedo_def'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_mask_albedo_def.png" for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_overexposure_mask'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_overexposure_mask.png" for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_specular_mask'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_specular_mask.png" for row in meta], subset=(start, end)))
            cols.append(Col('img',  '{}_mask'.format(name), [os.path.splitext(row[name]["pred_shading"])[0]+"_mask.png" for row in meta], subset=(start, end)))
            pass
        imagetable(cols, out_file="index_shading_{}_{}.html".format(suffix, i), imsize=[320, 213])
        pass
    
    # cols = [
    #     Col('id1',  'ID'),                                               # make a column of 1-based indices
    #     Col('img',  'input', meta_dict["img"]),             # specify image content for column 2
    #     Col('img',  'albedo gt', meta_dict["albedo_gt"]),             # specify image content for column 2
    #     Col('img',  'mask', meta_dict["mask"]),             # specify image content for column 2
        
    # ]
    # #         Col('img',  'ours_albedo', meta_dict["ours"]["albedo"]),             # specify image content for column 2
    # #     Col('text', 'ours_numbers', meta_dict["ours"]["number"]),     # specify image content for column 3

    # #     Col('img',  'ravi_albedo', meta_dict["ravi"]["albedo"]),             # specify image content for column 2
    # #     Col('text', 'ravi_numbers', ravi_numbers),     # specify image content for column 3

    # #     Col('img',  'cgintrinsics_albedo', meta_dict["cgintrinsics"]["albedo"]),             # specify image content for column 2
    # #     Col('text', 'cgintrinsics_numbers', cgintrinsics_numbers),     # specify image content for column 3

    # #     Col('img',  'bigtime_albedo', meta_dict["bigtime"]["albedo"]),             # specify image content for column 2
    # #     Col('text', 'bigtime_numbers', bigtime_numbers),     # specify image content for column 3
        
    # #     Col('img',  'soumyadip_albedo', meta_dict["soumyadip"]["albedo"]),             # specify image content for column 2
    # #     Col('text', 'soumyadip_numbers', soumyadip_numbers),     # specify image content for column 3        
    # for name in names:
    #     cols.append(Col('img',  '{}_albedo'.format(name), meta_dict[name]["albedo"]))
    #     cols.append(Col('img',  '{}_albedo_scaled'.format(name), meta_dict[name]["albedo_scaled"]))
    #     cols.append(Col('img',  '{}_albedo_gray'.format(name), meta_dict[name]["albedo_gray"]))
    #     cols.append(Col('img',  '{}_albedo_gray_scaled'.format(name), meta_dict[name]["albedo_gray_scaled"]))
    #     cols.append(Col('text', '{}_numbers'.format(name), meta_dict[name]["number"]))        
    #     pass
    
    # imagetable(cols, out_file="index_reduced_{}.html".format(suffix), imsize=[320, 240])

    
    pass


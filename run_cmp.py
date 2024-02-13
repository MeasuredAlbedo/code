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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="meta.csv")
    parser.add_argument("--type", type=str, default="metric", choices=["metric", "whdr", "shading"])
    parser.add_argument("--loss", type=str, default=numerical_albedo.LOSS_DEFAULT, choices=numerical_albedo.LOSS_CHOICES)
    parser.add_argument("--metric", type=str, default=numerical_albedo.METRIC_DEFAULT, choices=numerical_albedo.METRIC_CHOICES)
    args = parser.parse_args()
    mgr = utils.PathFinderMgr(args.meta)
    if args.type == "metric":
        evaluator = numerical_albedo.AlbedoEvaluator(loss=args.loss, metric=args.metric)
        pass
    elif args.type == "whdr":
        evaluator = numerical_albedo.WHDREvaluator()
        pass
    else:
        raise NotImplementedError(args.type)

    meta_dict = collections.defaultdict(lambda: [])
    names = ["ravi"]
    
    for row in mgr.iter_examples():
        color_lib = row.get_color_lib()
        mask = row.get_mask()
        img_path = row.get_img_path()
        if args.type =="whdr" and row.get_whdr_path() is None:
            continue
        meta_dict["img"].append(img_path)
        meta_dict["albedo_gt"].append(row.get_gt_albedo())
        meta_dict["mask"].append(mask)

        for name in names:
            #ours
            result = row.get_albedo(name)
            if name == "baseline":
                albedo, color_space = result, "srgb"
                pass
            else:
                albedo, color_space = result
                
            #log_our = run_ours(args.loss_str, color_lib, mask, ours_albedo)
            if args.type == "metric":
                metric, weights = evaluator.evaluate(color_lib, mask, albedo, color_space)
                pass
            elif args.type == "whdr":
                whdr = row.get_whdr_path()
                metric = evaluator.evaluate(whdr, albedo, srgb=True if color_space == "srgb" else False)
                pass
            else:
                raise NotImplementedError()
            
            if name not in meta_dict:
                meta_dict[name] = {}
                pass
            if "albedo_scaled" not in meta_dict[name]:
                meta_dict[name]["albedo_scaled"] = []
                pass
            if "albedo_gray" not in meta_dict[name]:
                meta_dict[name]["albedo_gray"] = []
                pass
            if "albedo_gray_scaled" not in meta_dict[name]:
                meta_dict[name]["albedo_gray_scaled"] = []
                pass
            
            meta_dict[name]["albedo_scaled"].append(evaluator.get_scaled_albedo_path(albedo))
            meta_dict[name]["albedo_gray"].append(evaluator.get_grayscale_albedo_path(albedo))
            meta_dict[name]["albedo_gray_scaled"].append(evaluator.get_grayscale_scaled_albedo_path(albedo))

            if "albedo" not in meta_dict[name]:
                meta_dict[name]["albedo"] = []
                pass
            
            meta_dict[name]["albedo"].append(albedo)

            if "log" not in meta_dict[name]:
                meta_dict[name]["log"] = []
                pass
            if args.type == "metric":
                meta_dict[name]["log"].append("deltaes: " + str(metric) + "weights: " + str(weights))
            #print(meta_dict[name]["log"])
            #print(log_our)
            if "number" not in meta_dict[name]:
                meta_dict[name]["number"] = []
                pass
        
            #meta_dict[name]["number"].append(metric.mean(axis=0))
            if args.type == "metric":
                meta_dict[name]["number"].append((metric * weights).sum() / weights.sum())
                pass
            else:
                meta_dict[name]["number"].append(metric.mean(axis=0))
                pass
            
            pass

        
        pass
    suffix = "{}_{}_{}_{}".format(os.path.splitext(os.path.basename(args.meta))[0], args.loss, args.metric, args.type)
    #output_csv = "output_{}.csv".format(suffix) if args.output_file is None else args.output_file
    output_csv = "output_{}.csv".format(suffix) 
    with open(output_csv, "wt") as f:
        writer = csv.writer(f)
        # for o,ravi, cg, bigtime, soumyadip in zip(*[meta_dict[name]["number"] for name in names]):
        #     writer.writerow([o, ravi, cg, bigtime, soumyadip])
        #     pass
        for fn, row in zip(meta_dict["img"], zip(*[meta_dict[name]["number"] for name in names])):
            dn = os.path.basename(os.path.dirname(fn))
            bn = os.path.basename(fn)
            writer.writerow(["{}_{}".format(dn, bn)]+list(row))
            pass

        pass
    
    #print(our_numbers)
    #print(ravi_numbers)
    cols = [
        Col('id1',  'ID'),                                               # make a column of 1-based indices
        Col('img',  'input', meta_dict["img"]),             # specify image content for column 2
        Col('img',  'albedo gt', meta_dict["albedo_gt"]),             # specify image content for column 2
        Col('img',  'mask', meta_dict["mask"]),             # specify image content for column 2
        
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
        cols.append(Col('img',  '{}_albedo'.format(name), meta_dict[name]["albedo"]))
        cols.append(Col('img',  '{}_albedo_scaled'.format(name), meta_dict[name]["albedo_scaled"]))
        cols.append(Col('text', '{}_log'.format(name), meta_dict[name]["log"]))        
        pass

    imagetable(cols, out_file="index_{}.html".format(suffix), imsize=[320, 240])

    cols = [
        Col('id1',  'ID'),                                               # make a column of 1-based indices
        Col('img',  'input', meta_dict["img"]),             # specify image content for column 2
        Col('img',  'albedo gt', meta_dict["albedo_gt"]),             # specify image content for column 2
        Col('img',  'mask', meta_dict["mask"]),             # specify image content for column 2
        
    ]
    #         Col('img',  'ours_albedo', meta_dict["ours"]["albedo"]),             # specify image content for column 2
    #     Col('text', 'ours_numbers', meta_dict["ours"]["number"]),     # specify image content for column 3

    #     Col('img',  'ravi_albedo', meta_dict["ravi"]["albedo"]),             # specify image content for column 2
    #     Col('text', 'ravi_numbers', ravi_numbers),     # specify image content for column 3

    #     Col('img',  'cgintrinsics_albedo', meta_dict["cgintrinsics"]["albedo"]),             # specify image content for column 2
    #     Col('text', 'cgintrinsics_numbers', cgintrinsics_numbers),     # specify image content for column 3

    #     Col('img',  'bigtime_albedo', meta_dict["bigtime"]["albedo"]),             # specify image content for column 2
    #     Col('text', 'bigtime_numbers', bigtime_numbers),     # specify image content for column 3
        
    #     Col('img',  'soumyadip_albedo', meta_dict["soumyadip"]["albedo"]),             # specify image content for column 2
    #     Col('text', 'soumyadip_numbers', soumyadip_numbers),     # specify image content for column 3        
    for name in names:
        cols.append(Col('img',  '{}_albedo'.format(name), meta_dict[name]["albedo"]))
        cols.append(Col('img',  '{}_albedo_scaled'.format(name), meta_dict[name]["albedo_scaled"]))
        cols.append(Col('img',  '{}_albedo_gray'.format(name), meta_dict[name]["albedo_gray"]))
        cols.append(Col('img',  '{}_albedo_gray_scaled'.format(name), meta_dict[name]["albedo_gray_scaled"]))
        cols.append(Col('text', '{}_numbers'.format(name), meta_dict[name]["number"]))        
        pass
    
    imagetable(cols, out_file="index_reduced_{}.html".format(suffix), imsize=[320, 240])

    
    pass


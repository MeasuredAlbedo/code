import csv
import os.path
import pathlib
class ALGORITHM_PATHS:
    raise RuntimeError("missing path to algorithms")
    IMGS_PATH = pathlib.Path()
    OURS_PATH = pathlib.Path()
    RAVI_PATH = pathlib.Path()
    CGINTRINSICS_PATH = pathlib.Path()
    BIGTIME_PATH = pathlib.Path()
    SOUMYADIP_PATH = pathlib.Path()
    USI3D_PATH = pathlib.Path()
    REVISIT_PATH = pathlib.Path()
    BELL2014_PATH = pathlib.Path()
    NIID_PATH = pathlib.Path()
    RETINEX_PATH = pathlib.Path()
    NESTMEYER_PATH = pathlib.Path()
    pass

class PathFinder:
    def __init__(self, row, imgs_dir=None): # mode
        # self.mode = mode
        self.color_lib, self.mask, self.pred_category, self.pred_name, self.gt_albedo, self.whdr_judgement = row
        self.rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "labels")
        self.imgs_dir = imgs_dir
        pass
    def get_img_path(self):
        imgs_path = ALGORITHM_PATHS.IMGS_PATH if self.imgs_dir is None else self.imgs_dir
        path = os.path.join(imgs_path, self.pred_category, self.pred_name) + ".png"
        return path
    
    def get_smooth_label_path(self):
        return self.get_mask().replace("_mask.png", "_smooth_mask.npy")

    def get_specular_mask_path(self):
        return self.get_mask().replace("_mask.png", "_specular_mask.png")

    def get_gt_shading_path(self):
        return self.get_mask().replace("_mask.png", "_shading.npy.npz")
    
    def get_gt_shading_blur_path(self):
        return self.get_mask().replace("_mask.png", "_shading_blur.npy.npz")
    
    def get_gt_shading_vis_path(self):
        return self.get_mask().replace("_mask.png", "_shading.png")
    
    def get_gt_shading_vis_blur_path(self):
        return self.get_mask().replace("_mask.png", "_shading_blur.png")
    
    def get_whdr_path(self):
        if self.whdr_judgement == "None":
            return None

        dataset, judgement = self.whdr_judgement.split("/")
        path = os.path.join(self.rel_path, dataset, judgement) + ".json"
        return path
    def get_color_lib(self):
        return os.path.join(self.rel_path, self.color_lib)

    def get_mask(self):
        return os.path.join(self.rel_path, self.mask)

    def get_gt_albedo(self):
        return os.path.join(self.rel_path, self.gt_albedo)
    
    def get_ours_albedo(self):
        path = os.path.join(os.path.join(ALGORITHM_PATHS.OURS_PATH, "eval_out/albedo_color/curr/epochs_50_dataset_scenenet_config_config_simple_v1_scenenet_lr_customized_no_reflection_enc_v3_in_affine_timestamp_1601945153", "albedo_"+self.pred_category+"_"+self.pred_name+".png"))
        return path, "srgb"

    def get_ravi_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.RAVI_PATH, "color_eval_cascade1_curr/results_brdf2_light10_brdf3_light10/", self.pred_name+"_"+"albedo1.png")
        return path, "srgb"

    def get_ravi_bs_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.RAVI_PATH, "color_eval_cascade1_curr/results_brdf2_light10_brdf3_light10/", self.pred_name+"_"+"albedoBS1.png")
        return path, "srgb"

    def get_ravi_iiw_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.RAVI_PATH, "color_eval_cascade1_iiw_curr/results_brdf2_light10", self.pred_name+"_"+"albedo0.png")
        return path, "srgb"
    
    def get_ravi_iiw_bs_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.RAVI_PATH, "color_eval_cascade1_iiw_curr/results_brdf2_light10", self.pred_name+"_"+"albedoBS0.png")
        return path, "srgb"

    def get_cgintrinsics_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.CGINTRINSICS_PATH, "eval_curr/", "albedo_linear_"+self.pred_category+"_"+self.pred_name+"_color.png")        
        # else:
        #     raise NotImplementedError(self.mode)
        return path, "srgb"

    # def get_cgintrinsics_albedo(self):
    #     path = os.path.join(ALGORITHM_PATHS.CGINTRINSICS_PATH, "result_ver6_test_aliasing/", "albedo_"+self.pred_category+"_"+self.pred_name+".png")        
    #     # else:
    #     #     raise NotImplementedError(self.mode)
    #     return path, "srgb"
    
    def get_cgintrinsics_filtered_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.CGINTRINSICS_PATH, "eval_curr/", "albedo_linear_"+self.pred_category+"_"+self.pred_name+"_guided_c3.0s45.0_color.png")
        return path, "srgb"

    def get_cgintrinsics_syn_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.CGINTRINSICS_PATH, "eval_syn_curr/", "albedo_"+self.pred_category+"_"+self.pred_name+".png")
        return path, "srgb"

    def get_bigtime_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.BIGTIME_PATH, "eval_curr/", "albedo_"+self.pred_category+"_"+self.pred_name+".png")
        return path, "srgb"

    def get_soumyadip_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.SOUMYADIP_PATH, "Data_curr/Sample/results/", self.pred_name+"_"+"albedo.png")
        return path, "linear"
    
    def get_soumyadip_syn_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.SOUMYADIP_PATH, "Data_syn_curr/Sample/results/", self.pred_name+"_"+"albedo.png")
        return path, "linear"

    def get_usi3d_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.USI3D_PATH, "results_curr/", self.pred_name, "output_r.png")
        return path, "linear"
    
    def get_revisit_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.REVISIT_PATH, "evaluation/results_curr/", "{}-R.png".format(self.pred_name))
        return path, "linear"
    
    def get_revisit_prime_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.REVISIT_PATH, "evaluation/results_curr/", "{}-r_prime.png".format(self.pred_name))
        return path, "linear"
    
    def get_bell2014_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.BELL2014_PATH, "curr/", "{}-r.png".format(self.pred_name))
        return path, "srgb"
    
    def get_NIID_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.NIID_PATH, "curr/decomposition_results/", "{}_decomposed_R.png".format(self.pred_name))
        return path, "srgb"

    def get_nonlocal_albedo(self):
        path = os.path.join(ALGORITHM_PATHS.RETINEX_PATH, "curr/", "{}_r.png".format(self.pred_name))
        return path

    def get_l1trans_albedo(self):
        path = os.path.join("/data/research/project/inverse_rendering_project/L1Flattening/ver6/", "{}_smooth.png".format(self.pred_name))
        return path

    def get_nestmeyer_albedo(self):
        path = os.path.join("/data/research/project/inverse_rendering_project/reflectance-filtering/results_curr", "{}-r_color.png".format(self.pred_name))
        return path, "srgb"
    def get_nestmeyer_filtered_albedo(self):
        path = os.path.join("/data/research/project/inverse_rendering_project/reflectance-filtering/results_curr", "{}-r_guided_c3.0s45.0_color.png".format(self.pred_name))
        return path, "srgb"

    # def get_flatten_albedo(self):
    #     path = os.path.join("/data/research/datasets/albedo_color/ver8_png_compress_test/", "{}.png".format(self.pred_name))
    #     return path

    def get_albedo(self, name):
        #print(name)
        if name == "ours":
            return self.get_ours_albedo()
        elif name == "ravi":
            return self.get_ravi_albedo()
        elif name == "ravi_bs":
            return self.get_ravi_bs_albedo()
        elif name == "ravi_iiw":
            return self.get_ravi_iiw_albedo()
        elif name == "ravi_iiw_bs":
            return self.get_ravi_iiw_bs_albedo()
        elif name == "cgintrinsics":
            return self.get_cgintrinsics_albedo()
        elif name == "cgintrinsics_syn":
            return self.get_cgintrinsics_syn_albedo()
        elif name == "cgintrinsics_filtered":
            return self.get_cgintrinsics_filtered_albedo()
        elif name == "bigtime":
            return self.get_bigtime_albedo()
        elif name == "soumyadip":
            return self.get_soumyadip_albedo()
        elif name == "soumyadip_syn":
            return self.get_soumyadip_syn_albedo()        
        elif name == "usi3d":
            return self.get_usi3d_albedo()
        elif name == "revisit":
            return self.get_revisit_albedo()
        elif name == "revisit_prime":
            return self.get_revisit_prime_albedo()
        elif name == "baseline":
            return self.get_img_path()
        elif name == "bell2014":
            return self.get_bell2014_albedo()
        elif name == "NIID":
            return self.get_NIID_albedo()
        elif name == "nonlocal":
            return self.get_nonlocal_albedo()
        elif name == "l1trans":
            return self.get_l1trans_albedo()
        elif name == "baseline_c":
            return name, "linear"
        elif name == "nestmeyer":
            return self.get_nestmeyer_albedo()
        elif name == "nestmeyer_filtered":
            return self.get_nestmeyer_filtered_albedo()
        elif name == "flatten":
            return self.get_flatten_albedo()
        else:
            raise NotImplementedError(name)
        pass

    def get_cgintrinsics_shading(self):
        path = os.path.join(ALGORITHM_PATHS.CGINTRINSICS_PATH, "eval_curr/", "shade_"+self.pred_category+"_"+self.pred_name+".png")        
        # else:
        #     raise NotImplementedError(self.mode)
        return path, "linear"

    # def get_cgintrinsics_albedo(self):
    #     path = os.path.join(ALGORITHM_PATHS.CGINTRINSICS_PATH, "result_ver6_test_aliasing/", "albedo_"+self.pred_category+"_"+self.pred_name+".png")        
    #     # else:
    #     #     raise NotImplementedError(self.mode)
    #     return path, "srgb"
    
    def get_bigtime_shading(self):
        path = os.path.join (ALGORITHM_PATHS.BIGTIME_PATH, "eval_curr/", "shade_"+self.pred_category+"_"+self.pred_name+".png")
        return path, "srgb"
 
    def get_usi3d_shading(self):
        path = os.path.join(ALGORITHM_PATHS.USI3D_PATH, "results_curr/", self.pred_name, "output_s.png")
        return path, "linear"
    
    def get_revisit_shading(self): #questionable
        path = os.path.join(ALGORITHM_PATHS.REVISIT_PATH, "evaluation/results_curr/", "{}-S.png".format(self.pred_name))
        return path, "linear"
    
    def get_bell2014_shading(self):
        path = os.path.join(ALGORITHM_PATHS.BELL2014_PATH, "curr/", "{}-s.png".format(self.pred_name))
        return path, "srgb"
    
    def get_NIID_shading(self):
        path = os.path.join(ALGORITHM_PATHS.NIID_PATH, "curr/decomposition_results/", "{}_decomposed_S.png".format(self.pred_name))
        return path, "srgb"

    def get_nestmeyer_shading(self): #questionable
        path = os.path.join("/data/research/project/inverse_rendering_project/reflectance-filtering/results_curr", "{}-s_colorized.png".format(self.pred_name))
        return path, "srgb"

    def get_soumyadip_shading(self):
        path = os.path.join(ALGORITHM_PATHS.SOUMYADIP_PATH, "Data_curr/Sample/results/", self.pred_name+"_"+"shading.png")
        return path, "srgb"

    def get_ravi_iiw_bs_shading(self):
        #path = os.path.join(ALGORITHM_PATHS.SOUMYADIP_PATH, "Data_curr/Sample/results/", self.pred_name+"_"+"shading.png")
        path = os.path.join(ALGORITHM_PATHS.RAVI_PATH, "color_eval_cascade1_iiw_curr/results_brdf2_light10", self.pred_name+"_"+"shading.png")
        return path, "srgb"

    def get_shading(self, name):
        #print(name)
        if name == "cgintrinsics":
            return self.get_cgintrinsics_shading()
        elif name == "bigtime":
            return self.get_bigtime_shading()
        elif name == "usi3d":
            return self.get_usi3d_shading()
        elif name == "revisit":
            return self.get_revisit_shading()
        elif name == "bell2014":
            return self.get_bell2014_shading()
        elif name == "NIID":
            return self.get_NIID_shading()
        elif name == "nestmeyer":
            return self.get_nestmeyer_shading()
        elif name == "soumyadip":
            return self.get_soumyadip_shading()
        elif name == "ravi_iiw_bs":
            return self.get_ravi_iiw_bs_shading()
        else:
            raise NotImplementedError()
        pass

    def get_low_freq_shading_mask_path(self):
        json_name = "{}.png.json".format(self.pred_name)
        full_path = os.path.join("/data/research/project/workspace/inverse_rendering/jupyters/albedo_measure/annotator_frequency/polygons_results/", json_name)
        return full_path
    def has_low_freq_shading_mask(self):
        return os.path.exists(self.get_low_freq_shading_mask_path())
    pass

class PathFinderMgr:
    def __init__(self, meta, imgs_dir=None):
        self.rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "labels")
        self.examples = []
        self.meta = meta
        with open(os.path.join(self.rel_path, self.meta)) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                self.examples.append(PathFinder(row, imgs_dir=imgs_dir))
                pass
            pass
        pass

    def iter_examples(self):
        for ex in self.examples:
            yield ex
        pass
    
    pass


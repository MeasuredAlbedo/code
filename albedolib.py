import rawpy
import imageio
#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mplPath
import json
from shapely.geometry import Polygon
import functools
import weakref
import os
import csv
import itertools
#import colormath.color_objects
#import colormath.color_diff
#from colormath.color_conversions import convert_color
import colour
from PIL import Image
import glob

def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())
def load_ral_colors(path):
    ral_dict = {}
    with open(path) as csvfile:
        ralreader = csv.reader(csvfile)
        ral_dict = {}
        next(ralreader)
        for row in ralreader:
            color_hex_str = row[2][1:]
            rgb = tuple(map(lambda x: int("".join(x), 16), chunk(color_hex_str, 2)))
            ral_dict[row[0]] = rgb
            pass
        pass
    return ral_dict

def srgb_to_linsrgb (srgb):
    """Convert sRGB values to physically linear ones. The transformation is
       uniform in RGB, so *srgb* can be of any shape.

       *srgb* values should range between 0 and 1, inclusively.

    """
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    return np.where (srgb > 0.04045, gamma, scale)

srgb2lin = srgb_to_linsrgb

def lin2srgb(lin):
    s1 = 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055
    s2 = 12.92 * lin
    s = np.where(lin > 0.0031308, s1, s2)
    return np.minimum(s, 1.0)

def lin2ravi(lin):
    return lin ** (1/ 2.2)

def ravi2lin(ravi):
    return ravi ** 2.2

def lin2adobergb(lin):
    return np.power(lin, 256/563)
def adobergb2lin(rgb):
    return np.power(rgb, 563/256)

def srgb2linadobe(srgb):
    adobe = colour.RGB_to_RGB(srgb,
                              colour.models.RGB_COLOURSPACE_sRGB,
                              colour.models.RGB_COLOURSPACE_ADOBE_RGB1998,
                              chromatic_adaptation_transform=None,
                              apply_cctf_decoding=True,
                              apply_cctf_encoding=False)
    #adobe = convert_color(colormath.color_objects.sRGBColor(*srgb.tolist()), colormath.color_objects.AdobeRGBColor)
    #return adobergb2lin(np.array(adobe))
    return np.clip(np.array(adobe), 0, 1)

def adobelin2srgb(lin):
    #adobergb = lin2adobergb(lin)
    #srgb = convert_color(colormath.color_objects.AdobeRGBColor(*adobergb.tolist()), colormath.color_objects.sRGBColor)
    srgb = colour.RGB_to_RGB(lin,
                             colour.models.RGB_COLOURSPACE_ADOBE_RGB1998,
                             colour.models.RGB_COLOURSPACE_sRGB,
                             chromatic_adaptation_transform=None,
                             apply_cctf_decoding=False,
                                     apply_cctf_encoding=True) 
    return np.clip(np.array(srgb), 0, 1)

def adobergb2srgb(rgb):
    #adobergb = lin2adobergb(lin)
    #srgb = convert_color(colormath.color_objects.AdobeRGBColor(*adobergb.tolist()), colormath.color_objects.sRGBColor)
    srgb = colour.RGB_to_RGB(rgb,
                             colour.models.RGB_COLOURSPACE_ADOBE_RGB1998,
                             colour.models.RGB_COLOURSPACE_sRGB,
                             chromatic_adaptation_transform=None,
                             apply_cctf_decoding=True,
                             apply_cctf_encoding=True) 
    return np.clip(np.array(srgb), 0, 1)

# def adobelin2hsvobj(lin):
#     adobergb = lin2adobergb(lin)
#     hsv = convert_color(colormath.color_objects.AdobeRGBColor(*adobergb.tolist()), colormath.color_objects.HSVColor)
#     return hsv

# def adobelin2hsv(lin):
#     #hsv_obj = adobelin2hsvobj(lin)
#     return np.array(hsv_obj.get_value_tuple())

# def adobelin2lab(lin):
#     adobergb = lin2adobergb(lin)
#     lab = convert_color(colormath.color_objects.AdobeRGBColor(*adobergb.tolist()), colormath.color_objects.LabColor)
#     return lab

# def srgb_to_lab_obj(srgb):
#     lab = convert_color(colormath.color_objects.sRGBColor(*srgb.tolist()), colormath.color_objects.LabColor)
#     return lab

def srgb_to_lab(srgb):
    #lab_obj = srgb_to_lab_obj(srgb)
    #return np.array(lab_obj.get_value_tuple())
    xyz = colour.sRGB_to_XYZ(srgb)
    lab = colour.XYZ_to_Lab(xyz)
    return lab
def srgb_to_hsv(srgb):
    hsv = convert_color(colormath.color_objects.sRGBColor(*srgb.tolist()), colormath.color_objects.HSVColor)
    return np.array(hsv.get_value_tuple())

def srgb_delta_e(srgb1, srgb2):
    lab1 = srgb_to_lab(srgb1/255)
    lab2 = srgb_to_lab(srgb2/255)
    return colormath.color_diff.delta_e_cie2000(lab1, lab2)

def adobelin_delta_e(lin1, lin2):
    lab1 = adobelin2lab(lin1)
    lab2 = adobelin2lab(lin2)
    return colormath.color_diff.delta_e_cie2000(lab1, lab2)
def gen_img(color, size=(32,32)):
    return np.full(list(size) + [3], color)
def gen_img_from_adobe_lin(color, size=(32, 32)):
    return gen_img(adobelin2srgb(color))

def lin_img_to_srgb(img):
    result = img.get_linear().copy()[::4, ::4]
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = adobelin2srgb(result[i, j])
            pass
        pass
    return result

def load_ral_colors_wiki(path):
    ral_dict = {}
    with open(path) as csvfile:
        ralreader = csv.reader(csvfile)
        ral_dict = {}
        next(ralreader)
        next(ralreader)
        for row in ralreader:
            #color_hex_str = row[2][1:]
            #rgb = tuple(map(lambda x: int("".join(x), 16), chunk(color_hex_str, 2)))
            r = int(row[5])
            g = int(row[6])
            b = int(row[7])
            
            #ral_dict[row[0]] = srgb_to_linsrgb(np.array([r, g, b]) / 255.0)
            ral_dict[row[0]] = (r, g, b)
            pass
        pass
    return ral_dict

def load_ral_colors_wiki_adobergb(path):
    ral_dict = {}
    with open(path) as csvfile:
        ralreader = csv.reader(csvfile)
        ral_dict = {}
        next(ralreader)
        next(ralreader)
        for row in ralreader:
            #color_hex_str = row[2][1:]
            #rgb = tuple(map(lambda x: int("".join(x), 16), chunk(color_hex_str, 2)))
            l = float(row[11])
            a = float(row[12])
            b = float(row[13])
            lab_color = colormath.color_objects.LabColor(l, a, b)
            adobe_rgb_color = convert_color(lab_color, colormath.color_objects.AdobeRGBColor)
            ral_dict[row[0]] = (adobe_rgb_color.clamped_rgb_r, adobe_rgb_color.clamped_rgb_g, adobe_rgb_color.clamped_rgb_b)
            pass
        pass
    return ral_dict

def _mask_for_img(img, points): #points: Nx2
    poly = mplPath.Path(points*np.array([[img.shape[1], img.shape[0]]]))
    coords = np.stack(np.mgrid[0:img.shape[1], 0:img.shape[0]],axis=0).transpose(1, 2, 0)
    mask = poly.contains_points(coords.reshape(-1, 2))
    return mask.reshape(img.shape[1], img.shape[0]).transpose()

def _find_robust_median(arr):
    return np.mean(arr.reshape(-1, 3), 0)

# def _find_robust_median(arr):
#     # t25 = np.quantile(arr, 0.25, axis=(0, 1), keepdims=True)
#     # t75 = np.quantile(arr, 0.75, axis=(0, 1), keepdims=True)
#     # filtered = arr[np.all(arr >= t25, axis=-1)]
#     # filtered = arr[np.all(arr <= t75, axis=-1)]
#     # med = np.mean(filtered, axis=0)

#     med = np.median(arr, axis=(0, 1)) # 3
#     err = np.linalg.norm(arr - med, axis=-1) #YxX
#     min_err_idx = np.argmin(err) # along whole
#     result = arr.reshape(-1, 3)[min_err_idx]
#     return result
        
# def _find_robust_median(arr):
#     arr = arr.reshape(-1, 3)
#     grayscale = arr.mean(axis=-1)
#     #arr = arr[np.isfinite(grayscale)]
#     #grayscale = grayscale[np.isfinite(grayscale)]
#     t25 = np.quantile(grayscale, 0.05, keepdims=True)
#     t75 = np.quantile(grayscale, 0.95, keepdims=True)
#     filtered = arr[np.logical_and(grayscale >= t25, grayscale <= t75)]

#     #print(arr.shape)
#     print('filtered', filtered.shape)
#     return np.mean(filtered, 0)
adobe_to_xyz = (
    0.57667, 0.18556, 0.18823, 0,
    0.29734, 0.62736, 0.07529, 0,
    0.02703, 0.07069, 0.99134, 0,
) # http://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf                                

xyz_to_srgb = (
    3.2406, -1.5372, -0.4986, 0,
    -0.9689, 1.8758, 0.0415, 0,
    0.0557, -0.2040, 1.0570, 0,
) # http://en.wikipedia.org/wiki/SRGB                                                     

def adobe_to_srgb(image):
    return image.convert('RGB', adobe_to_xyz).convert('RGB', xyz_to_srgb)

def is_adobe_rgb(image):
    return 'Adobe RGB' in image.info.get('icc_profile', '')

def dump_color_names(colors, path):
    with open(path, "wt") as f:
        for col in colors.keys():
            f.write("{}\n".format(col))
            pass
        pass
    pass
def dump_meta_color_libs(meta_path, category, label_dir, output_dir, colors):
    meta = open(meta_path, "wt")
    for lp in glob.glob(os.path.join(label_dir, "*.txt")):
        labels = open(lp).read().split()
        output_colors = []
        for l in labels:
            c = colors[l]
            output_colors.append(c)
            pass
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(lp))[0]+".npy")
        mask_path = os.path.join(output_dir, os.path.splitext(os.path.basename(lp))[0]+"_mask.png")
        np.save(output_path, output_colors)
        meta.write("{color}\t{mask}\t{category}\t{pred_name}\tNone\tNone\n".format(color=output_path, mask=mask_path, category=category, pred_name=os.path.splitext(os.path.basename(lp))[0]))
class ImageDesc(object):
    def __init__(self, path):
        self.path = path
        self.ann = None
        self.points = None
        self.color_guide_color = None
        self.color_guide_color_lin = None
        pass

    @memoized_method(maxsize=None)
    def get_linear(self, use_camera_wb=True, output_color=rawpy.ColorSpace.Adobe, user_wb=None):
        if user_wb is None:
            return rawpy.imread(self.path).postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, output_color=output_color, use_camera_wb=True, demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR) / 65535
        else:
            return rawpy.imread(self.path).postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, output_color=output_color, use_camera_wb=False, user_wb=user_wb, demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR) / 65535
            pass
        #return rawpy.imread(self.path).postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, output_color=output_color, use_camera_wb=False, use_auto_wb=False) / 65535
        #return rawpy.imread(self.path).postprocess(no_auto_bright=True, output_bps=16, output_color=output_color, use_camera_wb=use_camera_wb)
        #return rawpy.imread(self.path).postprocess(output_color=output_color, use_camera_wb=use_camera_wb, no_auto_bright=True)

        #return rawpy.imread(self.path).postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, output_color=rawpy.ColorSpace.sRGB, use_camera_wb=use_camera_wb) / 65535
        #print(self.path)
        # lin = adobergb2lin(imageio.imread(self.path) / 255)
        # img = Image.fromarray(np.round(lin*255).astype(np.uint8))
        # return np.array(adobe_to_srgb(img))/255
        pass
    def get_wb(self):
        im = rawpy.imread(self.path).postprocess(output_color=rawpy.ColorSpace.raw, gamma=(1, 1), use_camera_wb=False, output_bps=16, user_wb=[1.0, 1.0,1.0, 1.0], no_auto_bright=True, bright=1.0, demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)
        self.load_ann("annotator/polygon_results/{}.png.json".format(self.get_filename_no_ext()))
        mask = self.mask_for_img(im)
        wb = im[mask].mean(axis=0)
        norm_wb = np.array([wb[1]/wb[0], 1.0, wb[1]/wb[2], 1.0])
        return tuple(norm_wb.tolist())
    
    @memoized_method(maxsize=None)
    def get_vis(self, use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, user_wb=None):
        if user_wb is None:
            return rawpy.imread(self.path).postprocess(output_color=output_color, use_camera_wb=use_camera_wb, no_auto_bright=True)
        else:
            return rawpy.imread(self.path).postprocess(output_color=output_color, use_camera_wb=False, user_wb=user_wb, no_auto_bright=True)
        pass
    
    
    def get_path(self):
        return self.path
    
    def get_filename(self):
        return os.path.basename(self.get_path())

    def get_filename_no_ext(self):
        return os.path.splitext(self.get_filename())[0]
    
    def _points_from_ann(self, ann):
        points = []
        for p in ann["data"]:
            points.append((p["x"], p["y"]))
            pass
        
        return np.array(points)

    def load_ann(self, path):
        with open(path) as f:
            self.ann = json.load(f)
            self.points = self._points_from_ann(self.ann)
            pass
        pass

    def get_ann_points(self):
        return self.points
    
    def set_color_guide_color(self, color_guide_color):
        self.color_guide_color = np.array(color_guide_color)/255
        #self.color_guide_color_lin = srgb2lin(np.array(color_guide_color)/255)
        pass
    def set_color_guide_color_lin(self, color_guide_color_lin):
        self.color_guide_color_lin = np.array(color_guide_color_lin)
        pass

    def mask_for_img(self, img, points=None):
        if points is None:
            points = self.get_ann_points()
            pass
        
        return _mask_for_img(img, points)
        pass
    
    def color_of_masked_area(self, points=None):
        img = self.get_linear()
        mask = self.mask_for_img(img,points=points)
        masked = img[mask]
        med = _find_robust_median(masked)
        return med
    def get_color_guide_lin(self):
        # assert self.color_guide_color is not None
        # return srgb2linadobe(self.color_guide_color)
        return self.color_guide_color_lin
    
    def measured_by_reflectance(self):
        assert self.color_guide_color is not None
        measured = self.color_of_masked_area()
        return measured / self.get_color_guide_lin()
    # def obj_reflectance_alt(self):
    #     assert self.color_guide_color is not None
        
    pass

class ImageCollection(list):
    def __init__(self, flip_ref=False):
        super().__init__()
        self.ref_id = -1
        self.masked_ids = []
        self.masked_imgs = []
        self.flip_ref=flip_ref
        pass
    
    @staticmethod
    def from_seq(prefix, begin, end, **kwargs):
        imgs = ImageCollection(**kwargs)
        for i in range(begin, end+1):
            imgs.append(ImageDesc("{}/_DSC{}.ARW".format(prefix, i)))
            pass
        return imgs
    @staticmethod
    def from_lst(prefix, lst, **kwargs):
        imgs = ImageCollection(**kwargs)
        for l in lst:
            imgs.append(ImageDesc("{}/{}".format(prefix, l)))
            pass
        return imgs

    
    def mask_out_img(self, img_id):
        self.masked_imgs.append(self.pop(img_id))
        self.masked_ids.append(img_id)
        pass

    def clear_mask(self):
        assert len(self.masked_imgs) == self.masked_ids
        for mid, mim in zip(self.masked_ids, self.masked_imgs):
            self.insert(mid, mim)
            pass
        self.masked_ids.clear()
        self.masked_imgs.clear()
        pass
    
    def set_ref_id(self, ref_id):
        self.ref_id = ref_id
        pass

    def get_ref_img_desc(self):
        return self[self.ref_id]

    def get_views_desc(self):
        if not self.flip_ref:
            list = [im for i, im in enumerate(self) if i != self.ref_id]
            pass
        else:
            list = [im for i, im in enumerate(self) if i == self.ref_id]
            pass
        return list
    
    def map_views(self, func):
        descs = self.get_views_desc()
        result = [func(d) for d in descs]
        return result

    def map_all(self, func):
        result = [func(d) for d in self]
        return result

    def get_ref_img(self, img_type="linear"):
        ref_desc = self.get_ref_img_desc()
        if img_type=="linear":
            return ref_desc.get_linear()
        elif img_type=="vis":
            return ref_desc.get_vis()
        else:
            raise NotImplementedError()
        pass
    
    def get_views_img(self, img_type="linear"):
        if img_type == "linear":
            result = self.map_views(lambda v:v.get_linear())
            pass
        elif img_type == "vis":
            result = self.map_views(lambda v:v.get_vis())
            pass
        else:
            raise NotImplementedError()
        return result
    
    def residuals_downsampled(self, scale=65535, factor=4):
    #def residuals_downsampled(self, scale=255, factor=4):
        residuals = []
        img_ref = self.get_ref_img()
        imgs = self.get_views_img()
        
        ref_f = np.array(img_ref[::factor, ::factor, :], dtype=np.float)/scale
        for i, im in enumerate(imgs):
            im_f = np.array(im[::factor, ::factor, :], dtype=np.float)/scale
            residuals.append(np.linalg.norm(im_f - ref_f, ord=2, axis=-1))
            pass
        return residuals
    
    def write_for_annotator(self):
        try:
            wb = self[0].get_wb()
            pass
        except:
            wb = None
        for im in self.get_views_desc():
            im_name = im.get_filename_no_ext()
            imageio.imwrite("annotator/data/{}.png".format(im_name), im.get_vis(user_wb=wb), format="pillow", compress_level=1)
            pass
        pass

    def write_all_for_annotator(self):
        try:
            wb = self[0].get_wb()
            pass
        except:
            wb = None
            pass
        
        for im in self:
            im_name = im.get_filename_no_ext()
            imageio.imwrite("annotator/data/{}.png".format(im_name), im.get_vis(user_wb=wb), format="pillow", compress_level=1)
            pass
        pass
    
    def load_anns(self):
        for im in self.get_views_desc():
            im.load_ann("annotator/polygon_results/{}.png.json".format(im.get_filename_no_ext()))
            pass
        pass
    
    def all_view_polygons(self):
        polys = self.map_views(lambda d: d.get_ann_points())
        return polys
    
    def intersect_views_anns(self):
        pts = self.all_view_polygons()
        polys = []
        for p in pts:
            #print(p)
            polys.append(Polygon(p))
            pass
        result = polys[0]
        for p in polys[1:]:
            #print(result)
            result = result.intersection(p)
            pass
        #print(result)
        return list(result.exterior.coords)

    
    def find_best_match(self, regions=None):
        #residuals = self.residuals_downsampled()
        views = self.get_views_img()
        ref = self.get_ref_img()
        if regions is None:
            polys = self.all_view_polygons()
            pass
        else:
            polys = regions
            pass
        
        least_r = float('inf')
        idx = 0
        assert len(views) == len(polys)
        for i, (v, poly) in enumerate(zip(views, polys)):
            mask= _mask_for_img(v, poly)
            masked_v = v[mask]
            masked_r = ref[mask]
            # print(masked_v)
            # print(masked_r)
            med_v = _find_robust_median(masked_v)
            med_r = _find_robust_median(masked_r)
            res = np.linalg.norm(med_v - med_r, ord=2)
            med_v_scale = med_v / 65535
            med_r_scale = med_r / 65535
            print("med_v", med_v)
            print("med_r", med_r)
            med_v_adobe = lin2adobergb(med_v_scale)
            med_r_adobe = lin2adobergb(med_r_scale)
            
            med_v_srgb = convert_color(colormath.color_objects.AdobeRGBColor(*med_v_adobe.tolist()), colormath.color_objects.sRGBColor)
            med_v_srgb = np.array([med_v_srgb.clamped_rgb_r, med_v_srgb.clamped_rgb_g, med_v_srgb.clamped_rgb_b])
            med_r_srgb = convert_color(colormath.color_objects.AdobeRGBColor(*med_r_adobe.tolist()), colormath.color_objects.sRGBColor)
            med_r_srgb = np.array([med_r_srgb.clamped_rgb_r, med_r_srgb.clamped_rgb_g, med_r_srgb.clamped_rgb_b])
            
            print("med_v_srgb", med_v_srgb)
            print("med_r_srgb", med_r_srgb)
            print(res)
            print("----")
            if res < least_r:
                least_r = res
                idx = i
                pass
            pass
        return idx

    def load_ral_dict(self, path):
        self.ral_colors = load_ral_colors_wiki(path)
        pass

    def assign_ral_colors(self, *ral_name_list):
        views = self.get_views_desc()
        assert len(views) == len(ral_name_list)
        for v, name in zip(views, ral_name_list):
            print(self.ral_colors[name])
            v.set_color_guide_color(self.ral_colors[name])
            pass
        pass
    def obj_reflectance_alt(self):
        #wb_path = "{}/{}".format(prefix, lst[0].path)
        wb = ImageDesc(self[0].path).get_wb()
        #print("==========WARNING:applying sensor response hack=====================")
        mask = self[0].mask_for_img(self[0].get_linear(), self[1].get_ann_points())
        color_card_mask = self[0].get_linear(user_wb=wb)[mask] #- 0.0184 #hack1
        color_card_shading = color_card_mask / self[0].get_color_guide_lin()
        img_to_measure = self[1].get_linear(user_wb=wb)[mask] #- 0.0184 #hack2
        print(color_card_shading.shape)
        print(img_to_measure.shape)
        print(self[0].get_color_guide_lin())
        print(color_card_mask.min(axis=0), color_card_mask.max(axis=0))
        print(color_card_shading.min(axis=0), color_card_shading.max(axis=0))
        print(img_to_measure.min(axis=0), img_to_measure.max(axis=0))
       
        albedo = img_to_measure / color_card_shading
        print("img_to_measure:", img_to_measure.mean(axis=0))
        print("color_card_shading:", color_card_shading.mean(axis=0))
        print(albedo.min(axis=0), albedo.max(axis=0))
        return _find_robust_median(albedo)
    pass



def evaluate_consistency(colors):
    #compute mean & stddev in hsv separately
    hsv_colors = []
    for c in colors:
        hsv_colors.append(adobelin2hsv(c))
        pass
    hsv_colors = np.array(hsv_colors)
    mean = hsv_colors.mean(axis=0) #3
    variance = np.var(hsv_colors, axis=0, ddof=1) #
    stddev = np.sqrt(variance)
    return mean, variance, stddev

import rawpy
import imageio
import argparse
import os

import lensfunpy
import cv2
import subprocess
import imageio.v3 as imageio
import numpy as np
import pathlib
import matplotlib.pyplot as plt
def process_img(img_path, mode):
    
    print("img_path", img_path)

    raw = rawpy.imread(img_path)
    im_raw = raw.raw_image_visible
    # ext = os.path.splitext(img_path)[1]
    # if ext.lower() == ".arw":
    #     print(ext.lower())
    #     crop_origin_str = subprocess.run(["exiftool", "-DefaultCropOrigin", img_path, "-s3"], capture_output=True).stdout.decode("utf-8")
    #     crop_origin = np.array([int(s) for s in crop_origin_str.split()])
    #     print(crop_origin[0], crop_origin[1])
    # return

    
    # exit()

    # im = im[12:-12, 12:-12]
    cam_brand = subprocess.run(["exiftool", "-Make", img_path, "-s3"], capture_output=True).stdout.decode("utf-8").strip()
    cam_model = subprocess.run(["exiftool", "-Model", img_path, "-s3"], capture_output=True).stdout.decode("utf-8").strip()
    
    lens = subprocess.run(["exiftool", "-LensID", img_path, "-s3"], capture_output=True).stdout.decode("utf-8").strip()
    lens = "AF-S DX Zoom-Nikkor 18-55mm f/3.5-5.6G ED" if lens == "AF-S DX Zoom-Nikkor 18-55mm f/3.5-5.6G ED II" else lens
    print("cam_brand_str", cam_brand)
    print("cam_model", cam_model)
    print("lens", lens)
    db = lensfunpy.Database()
    cam = db.find_cameras(cam_brand, cam_model)[0]
    lens = db.find_lenses(cam, "", lens)[0]

    print("camera:", cam)
    print("lens:", lens)
    
    focal_complete = subprocess.run(["exiftool", "-FocalLength", img_path, "-s3"], capture_output=True)
    focal_str = focal_complete.stdout.decode("utf-8")
    focal_length = float(focal_str.replace("mm", "").strip())

    aperture_complete = subprocess.run(["exiftool", "-Aperture", img_path, "-s3"], capture_output=True)
    aperture_str = aperture_complete.stdout.decode("utf-8")
    aperture = float(aperture_str.replace("mm", "").strip())

    print("focal_length:", focal_length)
    print("aperture:", aperture)

    # color_pattern = raw.raw_pattern
    # color_desc = raw.color_desc.decode("utf-8")
    # pattern_letters = ''.join([color_desc[index] for index in color_pattern.flatten()])
    # if pattern_letters == "RGGB":
    #     comp_role = lensfunpy.LF_CR_RGGB
    #     pass
    # elif pattern_letters == "BGGR":
    #     comp_role = lensfunpy.LF_CR_BGGR
    #     pass
    # else:
    #     raise NotImplementedError(pattern_letters)
    # plt.imshow(im_raw)
    # comp_role = lensfunpy.LF_CR_RG
    # print("comp_role", comp_role)
    # mod_color = lensfunpy.Modifier(lens, cam.crop_factor, im_raw.shape[1], im_raw.shape[0])
    # mod_color.initialize(focal_length, aperture, pixel_format=np.uint16, flags=lensfunpy.ModifyFlags.ALL)
    # print(im_raw.itemsize)
    # print(im_raw.dtype)
    # im_raw_copy = im_raw.copy()
    # print("here2")
    # mod_color.apply_color_modification_bayers(im_raw_copy, comp_role=comp_role)
    # print("here1")
    # im_raw[:] = im_raw_copy
    # plt.figure()
    # plt.imshow(im_raw)
    # plt.show()
    
    # r = im_raw[::2, ::2]
    # r_shape = np.array([r.shape[1], r.shape[0]])
    # g1 = im_raw[::2, 1::2]
    # g1_shape = np.array([g1.shape[1], g1.shape[0]])
    # g2 = im_raw[1::2, ::2]
    # g2_shape = np.array([g2.shape[1], g2.shape[0]])
    # b = im_raw[1::2, 1::2]
    # b_shape = np.array([b.shape[1], b.shape[0]])


    #img_raw_shape = np.array([im_raw.shape[1], im_raw.shape[0]])
    # print(undist_coords.max(0).max(0).max(0), undist_coords.dtype)
    # undist_coords_r = (undist_coords[::2, ::2, 0, :]/(img_raw_shape-1)*(r_shape-1)).astype(np.float32)
    # undist_coords_g1 = undist_coords[::2, 1::2, 1, :].astype(np.float32)
    # undist_coords_g1[:, :, 0]-= 1
    # undist_coords_g1 = (undist_coords_g1 / (img_raw_shape-1)*(g1_shape-1)).astype(np.float32)
    # #((undist_coords[::2, 1::2, 1, :]-1)/(img_raw_shape-2)*(g1_shape-1)).astype(np.float32)
    # undist_coords_g2 = undist_coords[1::2, ::2, 1, :].astype(np.float32)
    # undist_coords_g2[:, :, 1] -= 1
    # undist_coords_g2 = (undist_coords_g2 / (img_raw_shape-1)*(g2_shape-1)).astype(np.float32)
    # #(undist_coords[1::2, ::2, 1, :]/(img_raw_shape-1)*(g2_shape-1)).astype(np.float32)
    # undist_coords_b = undist_coords[1::2, 1::2, 2, :].astype(np.float32)
    # undist_coords_b -= 1
    # undist_coords_b = (undist_coords_b / (img_raw_shape-1)*(b_shape-1)).astype(np.float32)

    # #(undist_coords[1::2, 1::2, 2, :]/(img_raw_shape-1)*(b_shape-1)).astype(np.float32)
    # mod = lensfunpy.Modifier(lens, cam.crop_factor, r_shape[0], r_shape[1])
    # mod.initialize(focal_length, aperture, pixel_format=np.uint16, flags=lensfunpy.ModifyFlags.ALL)
    # undist_coords_r = mod.apply_subpixel_geometry_distortion()[..., 0, :]
    # mod = lensfunpy.Modifier(lens, cam.crop_factor, g1_shape[0], g1_shape[1])
    # mod.initialize(focal_length, aperture, pixel_format=np.uint16, flags=lensfunpy.ModifyFlags.ALL)
    # undist_coords_g1 = mod.apply_subpixel_geometry_distortion()[..., 1, :]
    # mod = lensfunpy.Modifier(lens, cam.crop_factor, g2_shape[0], g2_shape[1])
    # mod.initialize(focal_length, aperture, pixel_format=np.uint16, flags=lensfunpy.ModifyFlags.ALL)
    # undist_coords_g2 = mod.apply_subpixel_geometry_distortion()[..., 1, :]
    # mod = lensfunpy.Modifier(lens, cam.crop_factor, b_shape[0], b_shape[1])
    # mod.initialize(focal_length, aperture, pixel_format=np.uint16, flags=lensfunpy.ModifyFlags.ALL)
    # undist_coords_b = mod.apply_subpixel_geometry_distortion()[..., 2, :]

    #rgb is just name
    # r_undistorted_raw = cv2.remap(r, undist_coords_r, None, cv2.INTER_CUBIC)
    # g1_undistorted_raw = cv2.remap(g1, undist_coords_g1, None, cv2.INTER_CUBIC)
    # g2_undistorted_raw = cv2.remap(g2, undist_coords_g2, None, cv2.INTER_CUBIC)
    # b_undistorted_raw = cv2.remap(b, undist_coords_b, None, cv2.INTER_CUBIC)
    
    # #im_raw_undistorte = np.stack([r_undistorted, g_undistorted, b_undistorted], axis=-1)
    # im_raw[::2, ::2]= r_undistorted_raw
    # im_raw[::2, 1::2]= g1_undistorted_raw
    # im_raw[1::2, ::2]= g2_undistorted_raw
    # im_raw[1::2, 1::2]= b_undistorted_raw
    
    if mode == "vis":
        im = raw.postprocess(no_auto_bright=True, output_color=rawpy.ColorSpace.sRGB, use_camera_wb=True, exp_shift=1.0, exp_preserve_highlights=0.0)
        pass
    elif mode == "full":
        im = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, output_color=rawpy.ColorSpace.Adobe, use_camera_wb=True)
        pass
    else:
        raise NotImplementedError(mode)
    mod = lensfunpy.Modifier(lens, cam.crop_factor, im.shape[1], im.shape[0])
    mod.initialize(focal_length, aperture, flags=lensfunpy.ModifyFlags.ALL)

    undist_coords = mod.apply_subpixel_geometry_distortion()
    r_undistorted = cv2.remap(im[:, :, 0], undist_coords[..., 0, :], None, cv2.INTER_CUBIC)
    g_undistorted = cv2.remap(im[:, :, 1], undist_coords[..., 1, :], None, cv2.INTER_CUBIC)
    b_undistorted = cv2.remap(im[:, :, 2], undist_coords[..., 2, :], None, cv2.INTER_CUBIC)

    im = np.stack([r_undistorted, g_undistorted, b_undistorted], axis=-1)

    ext = os.path.splitext(img_path)[1]
    if ext.lower() == ".arw":
        # crop_origin_str = subprocess.run(["exiftool", "-DefaultCropOrigin", img_path, "-s3"], capture_output=True).stdout.decode("utf-8")
        # crop_origin = np.array([int(s) for s in crop_origin_str.split()])
        # crop_size_str = subprocess.run(["exiftool", "-DefaultCropSize", img_path, "-s3"], capture_output=True).stdout.decode("utf-8")
        # crop_size = np.array([int(s) for s in crop_size_str.split()])
        # print("===FIXME====: crop_origin should be 1 to the bottom right. Unfortunately we have to fix latter")
        # top_left =crop_origin
        # bottom_right = crop_origin+crop_size
        
        #im = im[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        print("==fixme==: crop is just a hack")
        im = im[12:-12, 12:-12]
        pass
    print("im_shape:", im.shape)
    raw.close()
    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    # parser.add_argument("--cam_brand", type=str, default="SONY")
    # parser.add_argument("--cam_model", type=str, default="DSC-RX100M5A")
    parser.add_argument("--mode", type=str, choices=["vis", "full"], default="vis")
    args = parser.parse_args()
    
    #im_out = im_undistorted.astype(np.float32)/65525
    im_out = process_img(str(args.input), args.mode)
    #print(im_out.dtype)
    #output_path = args.input.with_suffix(".png")
    #print(output_path)
    imageio.imwrite(args.output, im_out, plugin="PNG-FI")
    pass

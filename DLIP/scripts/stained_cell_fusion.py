import os
import argparse
import tempfile
import yaml
import wandb
import logging

import tifffile
import cv2
import slidingwindow as sw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch 
import matplotlib
from skimage import measure
from skimage.draw import disk
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches
from skimage import measure
from skimage.color import label2rgb
from tqdm import tqdm

def get_dma_list(exp_dir, img_format=".tif"):
    dma_list = []

    for f in sorted(os.listdir(exp_dir)):
        ext = os.path.splitext(f)[1]  
        if ext.lower() != img_format:
            continue

        img_name = os.path.splitext("_".join(f.split("_")[-4:-1]))[0]
        dma_list.append(img_name)
    
    return list(set(dma_list))


if __name__ == "__main__":
    fusion_dirs   = ["Hoechst", "Calcein", "PI"]
    inst_ending = "_inst_seg"
    overlay_alpha = 0.3
    threshold = 6

    start_legend_x = 500
    start_legend_y = 2300

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_path',
        type=str,
        help='Path to data needs to be processed'
    )

    args, _ = parser.parse_known_args()

    dma_lst = get_dma_list(os.path.join(args.raw_data_path,fusion_dirs[0]))

    final_result_dir = os.path.join(args.raw_data_path, "final_results")

    if not os.path.exists(final_result_dir):
        os.makedirs(final_result_dir)

    for dma in tqdm(dma_lst, desc="Do stained cell fusion step"):
        data_dict = {ch: {"img_raw": None, "inst_seg": None}  for ch in fusion_dirs}
        for i, channel in enumerate(fusion_dirs):
            channel_dir = os.path.join(args.raw_data_path,channel)
            file_name = get_close_matches(dma, os.listdir(channel_dir),cutoff=0)[0]

            data_dict[channel]["img_raw"] = tifffile.imread(os.path.join(
                    channel_dir,
                    file_name)
                )

            data_dict[channel]["inst_seg"] = tifffile.imread(os.path.join(
                    channel_dir,
                    "results", "post_pro",
                    file_name.replace(".tif", f"{inst_ending}.tif"))
                )   


        ref_nuclei_props        = measure.regionprops(measure.label(data_dict[fusion_dirs[0]]["inst_seg"]))
        calcein_nuclei_props    = measure.regionprops(measure.label(data_dict["Calcein"]["inst_seg"]))
        pi_nuclei_props         = measure.regionprops(measure.label(data_dict["PI"]["inst_seg"]))
        ref_nuclei_lst          = list()
        calcein_nuclei_lst      = list()
        pi_nuclei_lst           = list()

        for prop in ref_nuclei_props:
            y, x = prop["centroid"]
            ref_nuclei_lst.append([x,y])

        for prop in calcein_nuclei_props:
            y, x = prop["centroid"]
            calcein_nuclei_lst.append([x,y])

        for prop in pi_nuclei_props:
            y, x = prop["centroid"]
            pi_nuclei_lst.append([x,y])

        neigh_calcein = NearestNeighbors(n_neighbors=1)
        neigh_calcein.fit(np.array(calcein_nuclei_lst))

        neigh_pi = NearestNeighbors(n_neighbors=1)
        neigh_pi.fit(np.array(pi_nuclei_lst))

        result_lst = list()

        for prop in ref_nuclei_props:
            y, x = prop["centroid"]
            point = np.array([x, y]).reshape(1,-1)

            dist_calcein, _ = neigh_calcein.kneighbors(point, return_distance=True)     
            dist_pi, _      = neigh_pi.kneighbors(point, return_distance=True)

            result_lst_i    = [x,y]

            # calculate intensities
            for ch in fusion_dirs:
                result_lst_i.append(np.median(data_dict[ch]["img_raw"][
                    data_dict["Hoechst"]["inst_seg"]==prop.label], axis=0))


            if dist_calcein<=threshold:
                result_lst_i.append(1)
                if dist_pi>threshold:
                    result_lst_i.append(1)
                else:
                    result_lst_i.append(0)
                result_lst.append(result_lst_i)

        results = np.array(result_lst)

        info = f"dma name: {dma}, contact: marcel.schilling@kit.edu (IAI) \n" \
            f"N_(hoechst/calcein): {results.shape[0]}, \n" \
            f"N_(hoechst/calcein+pi_neg): {np.sum(results[:,-1])}, \n" \
            f"x,y,med_int_Hoechst,med_int_Calcein,med_int_PI,Hoechst+/Calcein+,Hoechst+/Calcein+/PI-"
            
        output_dec_format = "%." + str(1) + "f"

        file_name = os.path.join(final_result_dir, f"{dma}_results_table.txt")
        np.savetxt(file_name, results, fmt=output_dec_format, delimiter=',', newline='\n', header=info, footer='',
                comments='# ')

        
        fontScale              = 0.8
        lineType               = 2
        font                   = cv2.FONT_HERSHEY_PLAIN

        image_lst = list()

        for ch in fusion_dirs:
            img_raw_i = ((data_dict[ch]["img_raw"]-np.min(data_dict[ch]["img_raw"]))/(np.max(data_dict[ch]["img_raw"]))-(np.min(data_dict[ch]["img_raw"])))*65535
            img_raw_i = cv2.cvtColor((img_raw_i/256).astype('uint8'), cv2.COLOR_GRAY2BGR)
            inst_seg  = label2rgb(data_dict[ch]["inst_seg"], bg_label=0)
            img_mask_i = (overlay_alpha*255*inst_seg+(1-overlay_alpha)*img_raw_i.astype("float32")).astype("uint8")
            img_detect_i = img_raw_i.copy()

            img_lst_i = [img_raw_i, img_mask_i, img_detect_i]

            for ind, img_i in enumerate(img_lst_i):
                img_i = np.pad(img_i, ((200, 200), (10, 10), (0,0)), 'constant', constant_values=((255, 255), (255,255), (0,0)))
                img_i = cv2.putText(img_i,f"Slide: {dma}, Channel {ch}", 
                    (0,120), 
                    font, 
                    4,
                    (0,0,0),
                    thickness=2
                )
                if ind==0:  
                    img_i = cv2.putText(img_i,f"+ : Hoechst pos, Calcein pos", 
                            (start_legend_x,start_legend_y), 
                            font, 
                            2,
                            (255,0,0),
                            thickness=2)

                    img_i = cv2.putText(img_i,f"+ : Hoechst pos, Calcein pos + PI neg", 
                            (start_legend_x,start_legend_y+30), 
                            font, 
                            2,
                            (0,255,0),
                            thickness=2)
                elif ind==1:
                    img_i = cv2.putText(img_i,f"Instance Segmentation Masks", 
                        (start_legend_x,start_legend_y), 
                        font, 
                        2,
                        (0,0,0),
                        thickness=2)
                else:
                    img_i = cv2.putText(img_i,f"Centroid Instance", 
                        (start_legend_x,start_legend_y), 
                        font, 
                        2,
                        (0,0,255),
                        thickness=2)


                img_i = cv2.putText(img_i,f"Reischl Lab (IAI)", 
                    (1650,50), 
                    font, 
                    3,
                    (0,0,0),
                    thickness=2)

                img_lst_i[ind] = img_i

            image_lst.append(img_lst_i)

        for res_i in results:
            if res_i[-1]==1:
                color = (0,255,0)
            else:
                color = (255,0,0)
            
            for j in range(3):
                image_lst[j][0] = cv2.drawMarker(image_lst[j][0], (int(res_i[0])+10,int(res_i[1])+200), color=color, markerType=cv2.MARKER_CROSS, thickness=1, markerSize=10)

        for point in ref_nuclei_lst:
            image_lst[0][-1] = cv2.drawMarker(image_lst[0][-1], (int(point[0])+10,int(point[1])+200), color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=1, markerSize=10)

        for point in calcein_nuclei_lst:
            image_lst[1][-1] = cv2.drawMarker(image_lst[1][-1], (int(point[0])+10,int(point[1])+200), color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=1, markerSize=10)

        for point in pi_nuclei_lst:
            image_lst[2][-1] = cv2.drawMarker(image_lst[2][-1], (int(point[0])+10,int(point[1])+200), color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=1, markerSize=10)

        image_final = np.stack((*image_lst[0],*image_lst[1], *image_lst[2]))
        tifffile.imwrite(os.path.join(final_result_dir,f"{dma}_results_overlay.tif"),image_final, imagej=True)
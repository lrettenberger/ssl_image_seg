import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2
import os
import skimage
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
from skimage import measure
from skimage.feature import peak_local_max, canny
from skimage.morphology import binary_closing
from skimage.color import label2rgb
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max

from sklearn.cluster import KMeans, DBSCAN

from copy import deepcopy

def get_nucleus_ids(img):
    """ Get nucleus ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


class InstSegProcessor:
    def __init__(self,
        th_cell=0.09,
        th_seed=0.49
    ) -> None:
        self.sigma_cell = 1.0
        self.th_cell    = th_cell
        self.th_seed    = th_seed
        self.do_splitting = True
        self.do_area_based_filtering = True
        self.do_fill_holes = True
        self.do_intensity_cluster_filtering =True
        self.do_intensity_dist_filtering = True
        self.valid_area_median_factors = [0.25,3]

    def process(self, pred_raw, img):
        pred = self._pre_process(pred_raw)
        pred_inst = self._pred2inst(pred)
        pred_inst_final = self._post_process(pred_inst, img)
        return pred_inst_final

    def _pre_process(self, raw_pred):  
        pred = gaussian_filter(raw_pred, sigma=self.sigma_cell)
        return pred

    def _pred2inst(self, pred):
        mask = pred > self.th_cell
        seeds =  pred > self.th_seed
        seeds = measure.label(seeds, background=0)
        if self.do_fill_holes:
            mask = binary_fill_holes(mask)
        pred_inst = watershed(image=-pred, markers=seeds, mask=mask, watershed_line=False)

        if self.do_splitting:
            pred_inst = self._perform_splitting(pred_inst, pred)

        return pred_inst

    def _post_process(self, pred_inst, img):
        if self.do_area_based_filtering:
            pred_inst = self._area_based_filter(pred_inst)

        if self.do_intensity_cluster_filtering:
            pred_inst = self._intensity_cluster_based_filter(pred_inst, img)

        if self.do_intensity_dist_filtering :
            pred_inst = self._intensity_dist_filter(pred_inst, img)
            
        return pred_inst

    def _area_based_filter(self,pred_inst): 
        area_lst = list()
        region_props = measure.regionprops(pred_inst)
        for prop in region_props:
            area_lst.append(prop.area)
        
        median_area = np.median(area_lst)

        for prop in region_props:
            if not (median_area*self.valid_area_median_factors[0] < prop.area < median_area*self.valid_area_median_factors[1]):
                pred_inst[pred_inst == prop.label] = 0

        pred_inst = measure.label(pred_inst, background=0)
        return pred_inst

    def _intensity_cluster_based_filter(self,pred_inst, img): 
        n_clusters = 6
        num_cluster2remove=1
        intensity_lst = list()
        region_props = measure.regionprops(pred_inst)
        for prop in region_props:
            intensity_lst.append(np.mean(img[pred_inst==prop.label]))


        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(intensity_lst).reshape(-1,1))
        cluster_assignment = kmeans.predict(np.array(intensity_lst).reshape(-1,1))

        # best_score = -np.inf

        # plt.hist(intensity_lst, bins=256)
        # plt.show()

        # for i_c in range(2,10):
        #     kmeans_i = KMeans(n_clusters=i_c, random_state=0).fit(np.array(intensity_lst).reshape(-1,1))
        #     cluster_assignment_i = kmeans_i.predict(np.array(intensity_lst).reshape(-1,1))
            
        #     if silhouette_score(np.array(intensity_lst).reshape(-1,1), cluster_assignment_i)>best_score:
        #         kmeans = deepcopy(kmeans_i)
        #         cluster_assignment = deepcopy(cluster_assignment_i)
        #         best_score = silhouette_score(np.array(intensity_lst).reshape(-1,1), cluster_assignment_i)

        
        #dbscan = DBSCAN(eps=(np.max(intensity_lst)-np.min(intensity_lst))/100, min_samples=3, metric='euclidean').fit(np.array(intensity_lst).reshape(-1,1))

        #cluster_assignment_db = dbscan.predict(np.array(intensity_lst).reshape(-1,1))

        #print(np.unique(dbscan.labels_))

       

        # plt.imshow(img, cmap="gray")
        # plt.show()

        removal_label_order = np.argsort(kmeans.cluster_centers_.squeeze())
        count = 0
        for ind,prop in enumerate(region_props):
            if cluster_assignment[ind] in removal_label_order[0:num_cluster2remove]:
                pred_inst[pred_inst == prop.label] = 0
                count +=1

        #print(count)
        pred_inst = measure.label(pred_inst, background=0)
        return pred_inst


    def _intensity_dist_filter(self,pred_inst, img): 
        region_props = measure.regionprops(pred_inst)
        count = 0
        for prop in region_props:
            mask = (pred_inst==prop.label)
            min_row, min_col, max_row, max_col = prop.bbox
            img_crop = img[min_row:max_row, min_col:max_col]
            try:

                peaks = peak_local_max(img_crop, num_peaks=3)

                dist = 0

                for i in range(3):
                    dist += np.sqrt((peaks[i][0]-img_crop.shape[0]/2)**2+(peaks[i][1]-img_crop.shape[1]/2)**2)

                dist = dist/(3*max(img_crop.shape[0]/2, img_crop.shape[1]/2))

                # print(dist)

                # print(prop.bbox)
                # thresh = threshold_otsu(img[min_row:max_row, min_col:max_col])
                # binary = img[min_row:max_row, min_col:max_col] > thresh

                # plt.imshow(peak_local_max(img[min_row:max_row, min_col:max_col], indices = False, num_peaks=3))
                # plt.plot( img_crop.shape[1]/2,  img_crop.shape[0]/2, "r*")
                # plt.show()

                # plt.imshow(img[min_row:max_row, min_col:max_col])
                # plt.show()

                if dist>0.5:
                    pred_inst[pred_inst == prop.label] = 0
            except:
                pass






            mask = cv2.erode(mask.astype("uint8"), np.ones((3,3)), iterations=2).astype("bool")
            try:
                # criteria = np.percentile(img[mask],95)/(np.max(img[mask])- np.min(img[mask]))
                # if criteria<0.825:
                #     count += 1
                #     pred_inst[pred_inst == prop.label] = 0
                kmeans = KMeans(n_clusters=2, random_state=0).fit(img[mask].reshape(-1,1))
                cluster_assignment = kmeans.predict(img[mask].reshape(-1,1))
                min_label = np.argmin(kmeans.cluster_centers_.squeeze())
                criteria = np.sum(cluster_assignment==min_label)/len(cluster_assignment)
                #print(criteria)
                if criteria>0.725:
                    count += 1
                    pred_inst[pred_inst == prop.label] = 0
            except:
                continue

        print(count)

        pred_inst = measure.label(pred_inst, background=0)
        return pred_inst

    def _perform_splitting(self, pred_inst, pred):
        props = measure.regionprops(pred_inst)
        volumes, nucleus_ids = [], []
        for i in range(len(props)):
            volumes.append(props[i].area)
            nucleus_ids.append(props[i].label)
        volumes = np.array(volumes)
        for i, nucleus_id in enumerate(nucleus_ids):
            if volumes[i] > np.median(volumes) + 2/5 * np.median(volumes):
                nucleus_bin = (pred_inst == nucleus_id)
                cell_prediction_nucleus = pred * nucleus_bin
                for th in [0.50, 0.60, 0.75]:
                    new_seeds = measure.label(cell_prediction_nucleus > th)
                    if np.max(new_seeds) > 1:
                        new_cells = watershed(image=-cell_prediction_nucleus, markers=new_seeds, mask=nucleus_bin,
                                              watershed_line=False)
                        new_ids = get_nucleus_ids(new_cells)
                        for new_id in new_ids:
                            pred_inst[new_cells == new_id] = np.max(pred_inst) + 1
                        break

        return pred_inst



if __name__ == "__main__":
    relevant_dirs   = ["Hoechst", "Calcein", "PI"]
    pred_raw_ending = "_pred_raw"
    inst_ending = "_inst_seg"
    overlay_ending = "_overlay_viz"
    overlay_alpha = 0.3

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data_path',
        type=str,
        help='Path to data needs to be processed'
    )

    args, _ = parser.parse_known_args()
    raw_dir = args.raw_data_path

    isp_obj= InstSegProcessor()

    for subdir in tqdm(os.listdir(args.raw_data_path), desc= f"Post-processing folder-wise"):
        if subdir in relevant_dirs:
            # if subdir != "Calcein":
            #     continue
            pred_raw_dir = os.path.join(args.raw_data_path, subdir ,"results", "pred_raw")
            post_pro_dir = os.path.join(args.raw_data_path, subdir, "results", "post_pro")

            if not os.path.exists(post_pro_dir):
                os.makedirs(post_pro_dir)

            for file in tqdm(os.listdir(os.path.join(raw_dir, subdir)),desc = "Post-processing of images"):
                if not os.path.isfile(os.path.join(raw_dir, subdir, file)):
                    continue

                img         = tifffile.imread(os.path.join(args.raw_data_path, subdir, file))
                pred_raw    = tifffile.imread(os.path.join(pred_raw_dir, file.replace(".tif", f"{pred_raw_ending}.tif")))

                pred_inst = isp_obj.process(pred_raw, img,)

                tifffile.imwrite(os.path.join(post_pro_dir, file.replace(".tif", f"{inst_ending}.tif")), pred_inst.astype("uint16"))

                img_norm = ((img-np.min(img))/(np.max(img))-(np.min(img)))*65535
                img_norm = cv2.cvtColor((img_norm/256).astype('uint8'), cv2.COLOR_GRAY2BGR)

                img_centroid = img_norm.copy()
                pred_rgb = label2rgb(pred_inst, bg_label=0)
                img_mask = (overlay_alpha*255*pred_rgb+(1-overlay_alpha)*img_norm.astype("float32")).astype("uint8")

                region_probs = measure.regionprops(measure.label(pred_inst, background=0))

                for prop in region_probs:
                    (row, col) = prop["centroid"]
                    pos = (int(col), int(row))
                    cv2.drawMarker(img_centroid, pos, color=(0,0,255), markerType=cv2.MARKER_CROSS, thickness=2)

                image_final = np.stack((img_centroid,img_mask, img_norm))

                tifffile.imwrite(os.path.join(post_pro_dir, file.replace(".tif", f"{overlay_ending}.tif")),image_final, imagej=True)

        # plt.imshow(img, cmap="gray")
        # plt.imshow(label2rgb(pred_inst, bg_label=0), alpha=0.1)
        # plt.show()
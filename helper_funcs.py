#functions for Leaf+Lesion_Classifier.py
#things that need fixing:
# - port to jupyter notebook
# - maybe make it so that you can use the classifiers separately?
import os 
import torch
from fastai import *
from fastai.vision import *
from timeit import default_timer as timer
from torch.nn import functional
import cv2 
import numpy as np
import skimage
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans
import scipy
import imageio
import fastai
from sklearn.preprocessing import Imputer
from PIL import Image
from UNet_init_funcs import *
import glob
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import re
#for testing purposes
import pickle 
from autoimpute.imputations import SingleImputer, MultipleImputer
import image_slicer

def Leaf_inference(leaf_learner_path, img):
    #load leaf U-NET model
    learn = load_learner(leaf_learner_path)
    #set to accomodate arbitrary input image size
    learn.data.single_dl.dataset.tfmargs['size'] = None
    #prediction yields mask image ([0]) and tensor ([1])
    prediction = learn.predict(img)
    return prediction

def slice_image(leaf_fpath, rescale_path):
    full_img = cv2.imread(leaf_fpath)
    dims = full_img.shape
    area = dims[0]*dims[1]
    n = int(area/(512*512))-1
    if n%2 != 0:
        n -= 1
    tiles = image_slicer.slice(leaf_fpath, n, save = False)
    image_slicer.save_tiles(tiles, directory = rescale_path)

def image_rescale(image, height, width, mode):
    if type(image) == str:
        img = cv2.imread(image)
        h,w,c = img.shape
    elif type(image) == np.ndarray:
        img = image
        h,w = image.shape

    if mode == 'fast':
        rescale_img = cv2.resize(img, (width, height))
    if mode == 'best' and height > h or width > w :
        rescale_img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    if mode == 'best' and height < h or width < w :
        rescale_img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    elif h == height and w == width:
        rescale_img = img
    return rescale_img

def dimensions_check_leaf(img_fname, ex_dir):
    #model requires dimensions to be even so checks and pads if necessary
    #probably save it to another name so that it doesn't remove image info 
    img_check = cv2.imread(img_fname)
    h,w,c = img_check.shape
    if h % 2 == 1:
        img_check = cv2.copyMakeBorder(img_check,1,0,0,0,cv2.BORDER_REPLICATE)
    if w % 2 == 1:
        img_check = cv2.copyMakeBorder(img_check,0,0,1,0,cv2.BORDER_REPLICATE)
    mod_img = img_check
    h2,w2,c2 = mod_img.shape
    if h2%2 == 0 and w2%2 == 0:
        cv2.imwrite(os.path.join(ex_dir,img_fname), mod_img)
    return mod_img

def dimensions_check_lesion(img_fname):
    #model requires dimensions to be even so checks and pads if necessary
    #probably save it to another name so that it doesn't remove image info 
    img_check = cv2.imread(img_fname)
    h,w,c = img_check.shape
    if h % 2 == 1:
        img_check = cv2.copyMakeBorder(img_check,1,0,0,0,cv2.BORDER_REPLICATE)
    if w % 2 == 1:
        img_check = cv2.copyMakeBorder(img_check,0,0,1,0,cv2.BORDER_REPLICATE)
    mod_img = img_check
    h2,w2,c2 = mod_img.shape
    if h2%2 == 0 and w2%2 == 0:
        #mod_path = os.path.join(mod_dir, img_fname)
        cv2.imwrite(img_fname, mod_img)
    return mod_img


def bbox_extraction(prediction_l,leaves_img):
    mask_image = prediction_l[0]
    mask_tensor = prediction_l[1]
    #need to rescale predicted tensor back up to original image size
    mask_np = mask_tensor.numpy()
    mask_np = np.uint8(mask_np.squeeze())
    h,w,c = leaves_img.shape
    mask_ori_size = image_rescale(mask_np, height = h, width = w, mode = 'best')

    #remove identified objects that 
    #label connected regions -- looks even better than contours?
    cleared = clear_border(mask_ori_size, bgval = 0)
    label_image = skimage.measure.label(cleared, background = 0, connectivity = 2)
    image_label_overlap = skimage.color.label2rgb(label_image, image = leaves_img)
    #fig,ax= plt.subplots(figsize = (10,6))
    #ax.imshow(image_label_overlap)
    #find the bounding boxes of each contour -- make it a bit bigger so there are slight borders?
    bounding_boxes = []
    for region in skimage.measure.regionprops(label_image):
    #filter rois by some kind of size? just to remove really small regions
        if region.area >= 2000:
            miny, minx, maxy, maxx = region.bbox
            bounding_boxes.append([miny, minx, maxy, maxx])
    #filter for a certain size? to remove really small 
    #size filter is troublesome... because really depends on the image... 
    #either train model better, or find some other way to remove artefacts
    return bounding_boxes, mask_ori_size

def leaf_kmeans_clustering(bounding_boxes, clust_by, number_of_classes):
    bb_df = pd.DataFrame(bounding_boxes)
    bb_df.columns = ['miny','minx','maxy','maxx']
    bb_df['mid_x'] = (bb_df['maxx']+bb_df['minx'])/2
    bb_df['mid_y'] = (bb_df['maxy']+bb_df['miny'])/2
    if clust_by == 'y':
        bb_df = bb_df.sort_values(by=['mid_y','mid_x'])
    elif clust_by == 'x':
        bb_df = bb_df.sort_values(by=['mid_x','mid_y'])
    #leaf_ID assigning -- 
    #K means to estimate number of rows, then rearrange individual rows -- noo wont work
    #Take group mid_y, and then clamp using mean leaf height?
    #estimate n rows by neighbor subtraction...

    bb_df['leaf_ID'] = ["%.2d" % i for i in range(1, len(bb_df)+1)]
    #cluster the mid-points... 
    kmeans = KMeans(n_clusters = number_of_classes, random_state = 26)
    kmeans_model = kmeans.fit(bb_df)
    centers = np.array(kmeans_model.cluster_centers_)

    if clust_by == 'y':
        labels = kmeans.fit_predict(bb_df[['mid_y']])
        bb_df['clust_group'] = labels
        sort_df = bb_df.sort_values(by='clust_group', ascending = True)
        lab_clas = sort_df.clust_group.unique()
        means = np.array(sort_df.groupby('clust_group')['mid_y'].mean())
        index = np.argsort(means)
        replace = dict(zip(index,lab_clas))
        new_labels = []
        for label in labels:
            new_labels.append(replace[label])
        bb_df['clust_group'] = new_labels
        #check the counts of each class
        clust_count = bb_df.groupby('clust_group')['leaf_ID'].nunique()

        #correct ID here... because will definitely be wrong most of the time 
        #get mean leaf height 
        ##############################################
        new_leaf_IDs = []
        for n in range(number_of_classes):
            #get clust_groups' coors
            clust = bb_df.loc[(bb_df['clust_group'] == n)]
            mean_leaf_height = np.mean(np.array(clust['maxy'])-np.array(clust['miny']))
            #get mid_ys of the leaves of the clust
            clust_mid_ys = np.array(clust['mid_y'])
            #should be in some sort of order already... 
            #neighbour subtraction is better, because shows how MANY rows
            y_diffs = []
            for y in range(len(clust_mid_ys)):
                if y != len(clust_mid_ys)-1:
                    diff = clust_mid_ys[y+1] - clust_mid_ys[y]
                    y_diffs.append(diff)
                elif y == len(clust_mid_ys)-1:
                    diff = 0
                    y_diffs.append(diff)
            #see if any y_diffs are really high 
            y_diffs = np.array(y_diffs)
            n_rows = np.sum(y_diffs>mean_leaf_height*0.5)+1
            #use k means to separate?
            if n_rows == 1:
                #take clust group and just clamp y to a fixed value, then order by x
                ori_IDs = np.array(clust['leaf_ID'])
                #y is considered clamped, so just look at mid_x
                clust_midx = np.array(clust['mid_x'])
                clust_order = np.argsort(clust_midx)
                ordered_IDs = np.zeros(ori_IDs.shape).astype(str)
                for i in range(len(clust_order)):
                    pos = clust_order[i]
                    val = ori_IDs[i]
                    ordered_IDs[pos] = val
                new_leaf_IDs.append(ordered_IDs)
            elif n_rows > 1:
                #define row_kmeans
                nested_new_IDs = []
                #mid_ys k
                row_kmeans = KMeans(n_clusters = n_rows, random_state = 26)
                row_labs = row_kmeans.fit_predict(clust_mid_ys.reshape(-1,1))
                ori_IDs = np.array(clust['leaf_ID'])
                clust_midx = np.array(clust['mid_x'])
                #each row is generally right... so just do the same as above for each row
                #row_labs will be random, so need to find someway to define upper and lower rows 
                for r in range(len(np.unique(row_labs))):
                    row = ori_IDs[row_labs == r]
                    row_midx = clust_midx[row_labs == r]
                    clust_order = np.argsort(row_midx)

                    ordered_row = np.zeros(row.shape).astype(str)
                    for i in range(len(clust_order)):
                        pos = clust_order[i]
                        val = row[i]
                        ordered_row[pos] = val
                    nested_new_IDs.append(ordered_row)
                
                #now nested_new_IDs will have the ordered rows, now need to arrange them in the right order
                nestlist_np= nested_new_IDs.copy()
                nestlist_np = [i.astype(int) for i in nestlist_np]
                nestlist_np = np.array(nestlist_np)

                #need to change to numbers b/c strings rn
                nest_means = []
                for l in nestlist_np:
                    m = np.mean(l)
                    nest_means.append(m)

                nest_means = np.array(nest_means)
                nest_order = np.argsort(nest_means)
                ordered_IDs = []
                for i in range(len(nest_order)):
                    ordered = nested_new_IDs[i]
                    ordered_IDs.append(ordered)
                
                #flatten list
                ordered_IDs = [item for sublist in ordered_IDs for item in sublist]
                new_leaf_IDs.append(ordered_IDs)
        #flatten new_leaf_IDs
        new_leaf_IDs = [item for sublist in new_leaf_IDs for item in sublist]
        bb_df['leaf_ID'] = new_leaf_IDs
        #####################################################

    if clust_by == 'x':
        labels = kmeans.fit_predict(bb_df[['mid_x']])
        bb_df['clust_group'] = labels
        sort_df = bb_df.sort_values(by='clust_group', ascending = True)
        lab_clas = sort_df.clust_group.unique()
        means = np.array(sort_df.groupby('clust_group')['mid_x'].mean())
        index = np.argsort(means)
        replace = dict(zip(index,lab_clas))
        new_labels = []
        for label in labels:
            new_labels.append(replace[label])
        bb_df['clust_group'] = new_labels
        #check the counts of each class
        clust_count = bb_df.groupby('clust_group')['leaf_ID'].nunique()

        new_leaf_IDs = []
        for n in range(number_of_classes):
            #get clust_groups' coors
            clust = bb_df.loc[(bb_df['clust_group'] == n)]
            mean_leaf_width = np.mean(np.array(clust['maxx'])-np.array(clust['minx']))
            #get mid_ys of the leaves of the clust
            clust_mid_xs = np.array(clust['mid_x'])
            #should be in some sort of order already... 
            #neighbour subtraction is better, because shows how MANY rows
            x_diffs = []
            for x in range(len(clust_mid_xs)):
                if x != len(clust_mid_xs)-1:
                    diff = clust_mid_xs[x+1] - clust_mid_xs[x]
                    x_diffs.append(diff)
                elif x == len(clust_mid_xs)-1:
                    diff = 0
                    x_diffs.append(diff)
            #see if any x_diffs are really high 
            x_diffs = np.array(x_diffs)
            n_cols = np.sum(x_diffs>mean_leaf_width*0.5)+1
            #use k means to separate?
            if n_cols == 1:
                #take clust group and just clamp y to a fixed value, then order by x
                ori_IDs = np.array(clust['leaf_ID'])
                clust_midy = np.array(clust['mid_y'])
                clust_order = np.argsort(clust_midy)
                ordered_IDs = np.zeros(ori_IDs.shape).astype(str)
                for i in range(len(clust_order)):
                    pos = clust_order[i]
                    val = ori_IDs[i]
                    ordered_IDs[pos] = val
                new_leaf_IDs.append(ordered_IDs)
            elif n_cols > 1:
                #define col_kmeans
                nested_new_IDs = []
                #mid_ys k
                col_kmeans = KMeans(n_clusters = n_cols, random_state = 26)
                col_labs = col_kmeans.fit_predict(clust_mid_xs.reshape(-1,1))
                ori_IDs = np.array(clust['leaf_ID'])
                clust_midy = np.array(clust['mid_y'])
                #each row is generally right... so just do the same as above for each row
                #row_labs will be random, so need to find someway to define upper and lower rows 
                for c in range(len(np.unique(col_labs))):
                    col = ori_IDs[col_labs == c]
                    col_midy = clust_midy[col_labs == c]
                    clust_order = np.argsort(col_midy)

                    ordered_col = np.zeros(col.shape).astype(str)
                    for i in range(len(clust_order)):
                        pos = clust_order[i]
                        val = col[i]
                        ordered_col[pos] = val
                    nested_new_IDs.append(ordered_col)
                
                #now nested_new_IDs will have the ordered rows, now need to arrange them in the right order
                nestlist_np= nested_new_IDs.copy()
                nestlist_np = [i.astype(int) for i in nestlist_np]
                nestlist_np = np.array(nestlist_np)

                #need to change to numbers b/c strings rn
                nest_means = []
                for l in nestlist_np:
                    m = np.mean(l)
                    nest_means.append(m)

                nest_means = np.array(nest_means)
                nest_order = np.argsort(nest_means)
                ordered_IDs = []
                for i in range(len(nest_order)):
                    ordered = nested_new_IDs[i]
                    ordered_IDs.append(ordered)
                
                #flatten list
                ordered_IDs = [item for sublist in ordered_IDs for item in sublist]
                new_leaf_IDs.append(ordered_IDs)
        #flatten new_leaf_IDs
        new_leaf_IDs = [item for sublist in new_leaf_IDs for item in sublist]
        bb_df['leaf_ID'] = new_leaf_IDs

    return bb_df

def bbox_measurements(bb_df):
    #make df that as area, height and width of each bbox 
    bbox_measurements = []
    for box in range(len(bb_df)):
        height = float(bb_df.iloc[box]['maxy']) - float(bb_df.iloc[box]['miny'])
        width = float(bb_df.iloc[box]['maxx'] - float(bb_df.iloc[box]['minx']))
        area = height*width
        bbox_measurements.append([height, width, area, id])

    measurements_df = pd.DataFrame(bbox_measurements)
    measurements_df.columns = ['height','width','area','ID']
    dfs = [bb_df, measurements_df]
    c_df = pd.concat(dfs, axis = 1)
    return c_df

def brightness_normalization(method, image, fname, norm_path, clip_hist_percent = 10):
    if method == 'CLAHE':
        #change to HSV color space
        leaf_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        leaf_l = leaf_lab[:,:,0]
        clahe = cv2.createCLAHE()
        hel = clahe.apply(leaf_l)
        leaf_lab[:,:,0] = hel
        leaf_bgr = cv2.cvtColor(leaf_lab, cv2.COLOR_LAB2BGR)
        bn_fname = fname[:-4] + '_' + method + '.png'
        bn_fpath = os.path.join(norm_path, bn_fname)
        cv2.imwrite(bn_fpath, leaf_bgr)
    
    if method == "alpha_beta":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''
        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        leaf_bgr = auto_result
        bn_fname = fname[:-4] + '_' + method + '.png'
        bn_fpath = os.path.join(norm_path, bn_fname)
        cv2.imwrite(bn_fpath, leaf_bgr)

    if method == None:
        pass

    return leaf_bgr


def bbox_means_sds(c_df):
    ms = []
    sds = []
    ns = ['height', 'width','area']
    for n in ns:
        ms.append(float(np.array(c_df[n].mean())))
        sds.append(float(np.array(c_df[n].std())))
    return ms, sds

def cluster_grouping_check(img_path, bb_df, leaf_path, mask_ori_size, alpha = 0.4):
    img = cv2.imread(img_path)
    for box in range(len(bb_df)):
        object_centre_x = int(bb_df.iloc[box]['mid_x'])
        object_centre_y = int(bb_df.iloc[box]['mid_y'])
        leaf_class = str(int(bb_df.iloc[box]['clust_group'] +1))
        txt_color = (255,0,0)
        cv2.putText(img, leaf_class, (object_centre_x-20, object_centre_y), cv2.FONT_HERSHEY_TRIPLEX,2,txt_color,6)
    img_fname = os.path.split(img_path)[1]
    fname_noext = img_fname[:-4]
    fig_fname = fname_noext + '_kmeans_grouping.jpg'
    fig_fname = os.path.join(leaf_path, fig_fname)
    beta = 1-alpha
    gamma = 0
    mask_layer = mask_ori_size*255
    h,w = mask_ori_size.shape
    mask_rgb = np.zeros((h,w,3))
    for channel in range(3):
        mask_rgb[:,:,channel] = mask_layer
    mask_rgb = np.uint8(mask_rgb)
    img = np.uint8(img)
    overlay = cv2.addWeighted(mask_rgb, alpha, img, beta, gamma)
    cv2.imwrite(fig_fname, overlay)

def leaf_id_check(img_path,bb_df, leaf_path, mask_ori_size, alpha = 0.4):
    img = cv2.imread(img_path)
    for box in range(len(bb_df)):
        object_centre_x = int(bb_df.iloc[box]['mid_x'])
        object_centre_y = int(bb_df.iloc[box]['mid_y'])
        leaf_id = str(bb_df.iloc[box]['leaf_ID'])
        txt_color = (255,0,0)
        cv2.putText(img, leaf_id, (object_centre_x-20, object_centre_y), cv2.FONT_HERSHEY_TRIPLEX,1.5,txt_color,6)
    img_fname = os.path.split(img_path)[1]
    fname_noext = img_fname[:-4]
    fig_fname = fname_noext + '_ids.jpg'
    fig_fname = os.path.join(leaf_path, fig_fname)
    beta = 1-alpha
    gamma = 0
    mask_layer = mask_ori_size*255
    h,w = mask_ori_size.shape
    mask_rgb = np.zeros((h,w,3))
    for channel in range(3):
        mask_rgb[:,:,channel] = mask_layer
    mask_rgb = np.uint8(mask_rgb)
    img = np.uint8(img)
    overlay = cv2.addWeighted(mask_rgb, alpha, img, beta, gamma)
    cv2.imwrite(fig_fname, overlay)

def potential_overlap_by_ids(overlaps, c_df,img_fname, src_path, dest_path, msk_path,msk_img):
    for o_ids in overlaps:
        leaf_id = str(o_ids)
        if len(leaf_id) == 1:
            leaf_id = '0' + leaf_id
        b = c_df.loc[c_df['leaf_ID'] == leaf_id]
        clust_group = int(np.array(b['clust_group']))
        leaf_class = str(clust_group+1)
        c_dir = 'Class_' + leaf_class
        leaf_fname = img_fname[:-4] + '_LEAF_' + str(leaf_id) +'.png'
        #locate leaf and then move it to potential overlap folder
        src = os.path.join(src_path,c_dir,leaf_fname)
        dest = os.path.join(dest_path, leaf_fname)
        shutil.copy(src,dest)
        #and move the mask!
        po_img = cv2.imread(src)
        h,w,c = po_img.shape
        xmin = int(np.array(b['minx'])) -10
        xmax = int(np.array(b['maxx'])) +10
        ymin = int(np.array(b['miny'])) -10
        ymax = int(np.array(b['maxy'])) +10
        msk_roi = msk_img[ymin:ymax, xmin:xmax]
        if msk_roi.shape[0:2] == po_img.shape[0:2]:
            msk_fname = leaf_fname[:-4] + '_mask.png'
            msk_p = os.path.join(msk_path,msk_fname)
            imageio.imwrite(msk_p, msk_roi*255)  
        elif msk_roi.shape[0:2] != po_img.shape[0:2]:
            raise Exception('mask shape is not equal to leaf image shape')
    
def potential_overlap_by_threshold(c_df, img_fname,ms,sds, src_path, dest_path, msk_path, msk_img):
    for box in range(len(c_df)):
    #threshold going to go by mean+2sd height OR width
        height = c_df.iloc[box]['height']
        width = c_df.iloc[box]['width']
        #can do area as well but can add in later 
        if height > (ms[0]+1.5*sds[0]) or width > (ms[1]+1.5*sds[1]):
            leaf_id = str(c_df.iloc[box]['leaf_ID'])
            clust_group = c_df.iloc[box]['clust_group']
            leaf_class = str(clust_group+1)
            c_dir = 'Class_' + leaf_class
            leaf_fname = img_fname[:-4] + '_LEAF_' + leaf_id +'.png'
            #locate leaf and then move it to potential overlap folder
            src = os.path.join(src_path, c_dir, leaf_fname)
            dest = os.path.join(dest_path, leaf_fname)
            shutil.copy(src,dest)
            #and move the mask!
            po_img = cv2.imread(src)
            xmin = int(c_df.iloc[box]['minx'] -10)
            xmax = int(c_df.iloc[box]['maxx'] +10)
            ymin = int(c_df.iloc[box]['miny'] -10)
            ymax = int(c_df.iloc[box]['maxy'] +10)
            msk_roi = msk_img[ymin:ymax, xmin:xmax]
            if msk_roi.shape[0:2] == po_img.shape[0:2]:
                msk_fname = leaf_fname[:-4] + '_mask.png'
                msk_p = os.path.join(msk_path,msk_fname)
                imageio.imwrite(msk_p, msk_roi*255)  
            elif msk_roi.shape[0:2] != po_img.shape[0:2]:
                raise Exception('mask shape is not equal to leaf image shape')

#####need to fix
def cmask_proc(c_msk_files, c_df, leaf_path, overlap_dir,po_files, t, clust_by):
    for file in c_msk_files:
        cmsk = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        #basic thresholding if user uses brush tool in photopea
        cmsk[cmsk != 255] = 0
        cmsk = (cmsk/255).astype(int)
        #objects touching borders are automatically removed - if need to compensate for sometimes
        #inference not being perfect then make 
        #bounding box a little bigger
        cmsk_cleared = clear_border(cmsk, bgval = 0)
        cmsk_ccs = skimage.measure.label(cmsk_cleared, background = 0, connectivity = 2)
        oid = 1 
        for region in skimage.measure.regionprops(cmsk_ccs):
        #filter rois by some kind of size? just to remove really small regions
            if region.area >= 500:
                miny, minx, maxy, maxx = region.bbox
                po_img_fname = po_files[0]
                po_img_fname = os.path.split(po_img_fname)[1]
                po_img_fname = po_img_fname.rsplit('_',1)[0]
                #add leaf number and extenstion
                leaf_number = str(re.findall(r'\d+', file)[-1])
                po_img_fname = po_img_fname + '_' + leaf_number + '.png'
                po_img_path = os.path.join(leaf_path, overlap_dir,po_img_fname)
                po_rgb = cv2.imread(po_img_path)
                new_roi = po_rgb[miny-5:maxy+5, minx-5:maxx+5]
                #need to get clust_group - for top bottom overlaps can be different dirs!
                #I think most direct is to map the roi back onto original image, depending on clust_by compare to 
                #c_df 
                #roi original coors
                df_ind = int(np.where(np.array(c_df['leaf_ID']) == leaf_number)[0])
                leaf_vals = np.array(c_df.iloc[df_ind])
                o_miny = leaf_vals[0]
                o_minx = leaf_vals[1]

                adj_miny, adj_minx, adj_maxy, adj_maxx = miny+o_miny, minx+o_minx, maxy+o_miny, maxx+o_minx

                if clust_by == 'y':
                    adj_mid_y = (adj_maxy + adj_miny)/2
                    clust_mid_ys = np.array(c_df.groupby('clust_group')['mid_y'].mean())
                    group_diffs = []
                    for v in clust_mid_ys:
                        abs_diff = abs(v - adj_mid_y)
                        group_diffs.append(abs_diff)
                    group_diffs = np.array(group_diffs)
                    clust_group = int(np.argmin(group_diffs))
                if clust_by == 'x':
                    adj_mid_x = (adj_maxx + adj_minx)/2
                    clust_mid_xs = np.array(c_df.groupby('clust_group')['mid_x'].mean())
                    group_diffs = []
                    for v in clust_mid_xs:
                        abs_diff = abs(v - adj_mid_x)
                        group_diffs.append(abs_diff)
                    group_diffs = np.array(group_diffs)
                    clust_group = int(np.argmin(group_diffs))

                c_df_id = str(leaf_number)
                o_clust_group = np.array(c_df.loc[c_df['leaf_ID'] == c_df_id]['clust_group'])
                o_c_dir = 'Class_' + str(int(clust_group)+1)
                roi_fname = os.path.split(file)[1]
                roi_fname = roi_fname[:-4] + '_' + str(oid) +  '.png'
                c_dir = 'Class_' + str(int(clust_group)+1)
                cv2.imwrite(os.path.join(leaf_path, c_dir, roi_fname), new_roi)
                oid = oid + 1   
        
        #little bit of code to find which overlap dir the deleted file should be in 
        res_dirs = glob.glob(leaf_path+'/Class_*')
        for d in res_dirs:
            fs = os.listdir(os.path.join(d))
            if po_img_fname in fs:
                rm_dir = d
        
        if t == 0: 
            os.remove(os.path.join(rm_dir,os.path.split(file)[1])) 
        elif t != 0:
            #yeah need to adjust file name to current image - can use po_image_fname
            os.remove(os.path.join(rm_dir,po_img_fname))


def Lesion_inference(lesion_learner_path, img):
    #load lesion U-NET model
    learn = load_learner(lesion_learner_path)
    learn.data.single_dl.dataset.tfmargs['size'] = None
    prediction = learn.predict(img)
    return prediction

def secondary_lesions_removal(mask_np,mask_fname, mask_dir):
    #clear up secondary lesion areas as leaf, then clean up the leaf pixels
    o_mask = mask_np.squeeze()
    o_mask_lp = (mask_np == 1) * 1
    o_mask_lesp = (mask_np == 2) * 1
    o_mask_bg = (mask_np == 0) * 1
    label_image = skimage.measure.label(o_mask,background = 0, connectivity = 2)
    ccs, size = np.unique(label_image, return_counts = True)
    sort_ind = np.argsort(-size)
    ccs_sort = ccs[sort_ind]
    #ccs_top = ccs_sort[0:3]
    h,w = mask_np.squeeze().shape
    label_image = np.array(label_image)

    leaf_labels = []
    lesion_labels = []
    #######leaf_label = area_sort.iloc[0]['label']
    for obj in range(len(ccs_sort)):
        obj_msk = (label_image == ccs_sort[obj]) * 1
        #want to see if the labelled object is originally leaf labelled or lesion labelled originally
        obj_leaf_mask = obj_msk + o_mask_lp
        obj_les_mask = obj_msk + o_mask_lesp
            
        obj_leaf_mask[obj_leaf_mask > 0] = 1
        obj_les_mask[obj_les_mask > 0] = 1

        obj_leaf_sum = np.sum(obj_leaf_mask)
        obj_les_sum = np.sum(obj_les_mask)
        leaf_sum = np.sum(o_mask_lp)
        lesion_sum = np.sum(o_mask_lesp)

        if obj_leaf_sum == leaf_sum:
            leaf_labels.append(ccs_sort[obj])
        elif obj_les_sum == lesion_sum:
            lesion_labels.append(ccs_sort[obj])
        else:
            pass
        #find the bounding boxes of each contour -- make it a bit bigger so there are slight borders?
    cc_props = []
    h,w = label_image.shape
    mid_x = w/2
    mid_y = h/2
    for region in skimage.measure.regionprops(label_image):
    #filter rois by some kind of size? just to remove really small regions
        miny, minx, maxy, maxx = region.bbox
        centre_x = maxx-minx
        centre_y = maxy-miny
        dist_centre = int((mid_x - centre_x)**2 + (mid_y - centre_y)**2)
        label = region.label
        area = region.convex_area
        cc_props.append([dist_centre,label,area])
                
    props_df = pd.DataFrame(cc_props,columns = ['dist_from_centre','label','area'])
    #can't assume leaf labels are the most numerous -- 
    #need some other way of identifying leaf label
    area_sort = props_df.sort_values(by = 'area', ascending = False)
    #create a list of potential leaf labels and potential lesion labels
    #ccs CAN be greater than 3 in some cases - (if lesion cuts between two leaf sections)
    #so more generalizable to correct if number of possible LESION labels > 
    

    if len(lesion_labels)>1:
        if len(np.unique(o_mask)) > 2:
            #drop all the leaf labels in the df  
            dist_sort_les = area_sort[~area_sort['label'].isin(leaf_labels)]
            dist_sort_les = dist_sort_les.sort_values(by = ['dist_from_centre'],ascending = [True])
            lesion_label = dist_sort_les.iloc[0]['label']
            area_sort_leaf = area_sort[area_sort['label'].isin(leaf_labels)]
            #another definition of leaf label - the labels attached to largest lesion
            potential_leaf_ids = [l for l in ccs_sort if l not in [0, lesion_label]]
            leaf_bool = []
            for Lid in potential_leaf_ids:
                largest_ls_mask = (label_image == lesion_label)*1
                pot_leaf_mask = (label_image == Lid)*1
                ls_pot_mask = largest_ls_mask + pot_leaf_mask
                pl_ccs = skimage.measure.label(ls_pot_mask, background = 0, connectivity = 2)
                if len(np.unique(pl_ccs)) == 2:
                    leaf_bool.append(1)
                elif len(np.unique(pl_ccs)) > 2:
                    leaf_bool.append(0)
            
            leaf_labels = [potential_leaf_ids[I] for I in range(len(leaf_bool)) if leaf_bool[I] == 1]

            #leaf_label_1 = area_sort_leaf.iloc[0]['label']
            #leaf_labels = [leaf_label_1]
        if len(np.unique(o_mask)) <= 2:
            lesion_label = 2
            leaf_label_1 = 1
            leaf_labels = [leaf_label_1]

        not_in_list = [0, lesion_label] + leaf_labels
        label_ids = [l for l in ccs_sort if l not in not_in_list]
        leaf_masks = []
        #corrected_mask = label_image.copy()
        replace_vals = []
        #ccs_top[0] = leaf pixels - 
        #can make sure of that by calculating perimeter of the values i guess?
        #can also take location information... e.g. take centre value of image and so on
        #can see use bbox info to see what should be assigned i guess.
        for o in range(len(label_ids)):
            leaf_area_mask = o_mask_lp
            object_area_mask = (label_image == label_ids[o])*1
            leaf_object_mask = leaf_area_mask + object_area_mask
            mask_ccs = skimage.measure.label(leaf_object_mask, background = 0, connectivity = 2)

            if len(np.unique(mask_ccs)) <= 3:
                replace_vals.append(1)
            elif len(np.unique(mask_ccs) > 3):
                replace_vals.append(0)
                    
        replace_dict = dict(zip(label_ids, replace_vals))
        bg = {0:0}
        leaf_label = 1
        l = {**dict.fromkeys(leaf_labels, leaf_label)}
        primary_les = {lesion_label: 2}
        updates =[bg,l, primary_les]
        for ud in updates:
            replace_dict.update(ud)
        replace_func = np.vectorize(lambda x: replace_dict[x])
        corrected_mask = replace_func(label_image)
        lesion_label = 2
        img_state = 3
        
        #FINAL correction 
        #plus shouldnt use ccs_sort as a check - b/c it's before correction
        #secondary correction to remove satellite lesions

    elif len(lesion_labels) == 1:
        #can't be just equal to mask_np b/c doesn't have same label numbers
        corrected_mask = label_image
        area_sort_leaf = area_sort[area_sort['label'].isin(leaf_labels)]
        not_lesions = leaf_labels.copy()
        not_lesions.append(0)
        les_area_sort = area_sort[~area_sort['label'].isin(not_lesions)]
        lesion_label = les_area_sort.iloc[0]['label']        #what if there is more than one leaf label?
        potential_leaf_ids = [l for l in ccs_sort if l not in [0, lesion_label]]
        leaf_bool = []
        for Lid in potential_leaf_ids:
            largest_ls_mask = (label_image == lesion_label)*1
            pot_leaf_mask = (label_image == Lid)*1
            ls_pot_mask = largest_ls_mask + pot_leaf_mask
            pl_ccs = skimage.measure.label(ls_pot_mask, background = 0, connectivity = 2)
            if len(np.unique(pl_ccs)) == 2:
                leaf_bool.append(1)
            elif len(np.unique(pl_ccs)) > 2:
                leaf_bool.append(0)
            
        leaf_labels = [potential_leaf_ids[I] for I in range(len(leaf_bool)) if leaf_bool[I] == 1]
        replace_dict = {**dict.fromkeys(leaf_labels,1), lesion_label: 2, 0:0}
        not_in_list = [lesion_label,0] + leaf_labels
        for l in ccs_sort:
            if l not in not_in_list:
                ud = {l:0}
                replace_dict.update(ud)
        replace_func = np.vectorize(lambda x: replace_dict[x])
        corrected_mask = replace_func(corrected_mask)
        leaf_label = 1
        lesion_label = 2
        img_state = 3

    elif len(lesion_labels) == 0:
        corrected_mask = o_mask
        leaf_label = 1
        lesion_label = 2
        img_state = 2
    return corrected_mask, leaf_label, lesion_label, img_state, label_image, leaf_labels

def leaf_measures_calculations(fname, bn_dir,corrected_mask,img_state, tgi_params, leaf_label, lesion_label):
    lambdaR = tgi_params[0]
    lambdaG = tgi_params[1]
    lambdaB = tgi_params[2]
    if img_state == 3:
    #before is if check_ccs >=3:
    #leaf size
        vals, freqs = np.unique(corrected_mask, return_counts = True)
        leaf_pn = freqs[np.where(vals == leaf_label)]
        lesion_pn = freqs[np.where(vals == lesion_label)]

        #now get RGB values from original image
        if bn_dir == None:
            img_bgr = cv2.imread(fname)
        elif bn_dir != None:
            bn_path = os.path.join(os.path.split(fname)[0],bn_dir,os.path.split(fname)[1])
            img_bgr = cv2.imread(bn_path)
        leaf_pix = (corrected_mask == leaf_label)*1
        leaf_bgr = img_bgr.copy()
        for ax in range(3):
            leaf_bgr[:,:,ax] = leaf_bgr[:,:,ax] * leaf_pix

        #calculate leaf greenness index? or use triangular greenness index
        leaf_b = leaf_bgr[:,:,0]/255
        leaf_b_sum = np.sum(leaf_b)
        leaf_g = leaf_bgr[:,:,1]/255
        leaf_g_sum = np.sum(leaf_g)
        leaf_r = leaf_bgr[:,:,2]/255
        leaf_r_sum = np.sum(leaf_r)

        Lesion_size = float(lesion_pn/(lesion_pn+leaf_pn))
        TGI = float(np.sum(0.5*((lambdaR - lambdaB)*(leaf_r - leaf_g) - (lambdaR - lambdaG)*(leaf_r - leaf_b)))/leaf_pn)
        NGRDI = float(((leaf_g_sum - leaf_r_sum)/(leaf_g_sum + leaf_r_sum))*255)
        lsn = float(lesion_pn)

    elif img_state == 2:
        #now get RGB values from original image 
        if bn_dir == None:
            img_bgr = cv2.imread(fname)
        elif bn_dir != None:
            img_bgr = cv2.imread(os.path.join(bn_dir, fname))
        leaf_pix = (corrected_mask == leaf_label)*1
        leaf_pn = np.sum(leaf_pix)
        leaf_bgr = img_bgr.copy()
        for ax in range(3):
            leaf_bgr[:,:,ax] = leaf_bgr[:,:,ax] * leaf_pix
        #calculate leaf greenness index? or use triangular greenness index
        leaf_b = leaf_bgr[:,:,0]/255
        leaf_b_sum = np.sum(leaf_b)
        leaf_g = leaf_bgr[:,:,1]/255
        leaf_g_sum = np.sum(leaf_g)
        leaf_r = leaf_bgr[:,:,2]/255
        leaf_r_sum = np.sum(leaf_r)

        #TGI i guess? using default values from original paper

        Lesion_size = float(0)
        TGI = float(np.sum(0.5*((lambdaR - lambdaB)*(leaf_r - leaf_g) - (lambdaR - lambdaG)*(leaf_r - leaf_b)))/leaf_pn)
        NGRDI = float(((leaf_g_sum - leaf_r_sum)/(leaf_g_sum + leaf_r_sum))*255)
        lsn = float(0)

    return Lesion_size,TGI, NGRDI, lsn

def stats_analysis(leaf_measures,stats_test):
    anova_p_vals = []
    #p_vals adjusted using bonferroni correction
    tukeys_res = []
    group_means = []
    for measure in leaf_measures:
        if stats_test == 'anova':
            statistic, pvalue = scipy.stats.f_oneway(*measure)
            anova_p_vals.append(float(pvalue*len(leaf_measures)))
            if pvalue < 0.05:
                #change measures to df
                df = pd.DataFrame(measure)
                df.index = [str(i+1) for i in range(len(df.index))]
                df = df.transpose()
                #impute NaNs - mean of column?
                df = df.fillna(df.mean())
                df_melt = df.melt()
                mc = MultiComparison(df_melt['value'].astype('float'),df_melt['variable'])
                res = mc.tukeyhsd().summary()
                tukeys_res.append(res)
                group_means.append(np.array(np.mean(df)))

            else:
                tukeys_res.append('Null')
                df = pd.DataFrame(measure)
                df = df.transpose()
                df = df.fillna(df.mean())
                group_means.append(np.array(np.mean(df)))
    return anova_p_vals, tukeys_res, group_means

def mask_gap_fill(corrected_mask, kernel_size = 7):
    #create empty np array of the same size 
    corrected_mask = np.uint8(corrected_mask)
    h,w = corrected_mask.shape
    filled_mask = np.zeros((h,w), np.uint8)
    kernel = np.ones((kernel_size, kernel_size),np.uint8)
    for i in range(len(np.unique(corrected_mask))):
        layer = (corrected_mask == i+1)*1
        layer = np.uint8(layer)
        layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel)
        filled_mask += (layer*i+1)
    if len(np.unique(filled_mask)) > 3:
        filled_mask[filled_mask > 2] = 2
    return filled_mask

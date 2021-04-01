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
import UNet_init_funcs
import glob
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from helper_funcs import *
import re
from autoimpute.imputations import SingleImputer, MultipleImputer

def Run_Leaf_Inference(leaf_fpath, clust_by, number_of_classes, time_series_analysis, use_t0_as_reference, overlap_ids, out_dir, bright_norm,t,t0_fpath, leaf_learner_path = 'Leaf_UNet/v1', size = 64,  stats_test = 'anova'):

    number_of_classes = number_of_classes
    img_path = leaf_fpath
    img_fname = os.path.split(img_path)[1]
    ex_dir = out_dir
    if not os.path.exists(ex_dir):
        os.mkdir(ex_dir)

    #check image dimensions
    rescale_dir = 'rescaled_images'
    rescale_path = os.path.join(ex_dir, rescale_dir)
    if not os.path.exists(rescale_path):
        os.mkdir(rescale_path)

    img_512 = image_rescale(image = img_path, height = 512, width = 512, mode = "best")
    cv2.imwrite(os.path.join(rescale_path, img_fname), img_512)
    img = open_image(os.path.join(rescale_path,img_fname))
    #inference
    if time_series_analysis != 'y':
        prediction_l = Leaf_inference(leaf_learner_path, img)
    elif time_series_analysis == 'y' and t == 0:
        prediction_l = Leaf_inference(leaf_learner_path, img)
    elif time_series_analysis == 'y' and t != 0 and use_t0_as_reference == 'n':
        prediction_l = Leaf_inference(leaf_learner_path, img)
    elif time_series_analysis == 'y' and t != 0 and use_t0_as_reference == 'y':
        #just tell it to predict on the same image as t0
        t0_fname = os.path.split(t0_fpath)[1][:-4] + img_fname[-4:]
        t0_img = open_image(os.path.join(rescale_path, t0_fname))
        prediction_l = Leaf_inference(leaf_learner_path, t0_img)

    #scale the prediction and image back up then bbox_extract
    leaves_img = cv2.imread(img_path)
    if time_series_analysis == 'n':
        bounding_boxes, mask_ori_size = bbox_extraction(prediction_l, leaves_img)
    elif time_series_analysis == 'y' and use_t0_as_reference == 'n':
        bounding_boxes, mask_ori_size = bbox_extraction(prediction_l, leaves_img)
    elif time_series_analysis == 'y' and use_t0_as_reference == 'y' and t == 0:
        bounding_boxes, mask_ori_size = bbox_extraction(prediction_l, leaves_img)
    elif time_series_analysis == 'y' and use_t0_as_reference == 'y' and t != 0:
        bounding_boxes, mask_ori_size = bbox_extraction(prediction_l, leaves_img)

    fname_noext = os.path.splitext(img_fname)[0]
    leaves_dir = fname_noext + '_res'
    leaves_dir = os.path.join(ex_dir, leaves_dir)

    if not os.path.exists(leaves_dir):
        os.mkdir(leaves_dir)

    #save the segmentation mask img
    mask_fname = fname_noext + '_mask.png'
    imageio.imwrite(os.path.join(leaves_dir,mask_fname), mask_ori_size*255)
    leaf_path = leaves_dir

    #OKAY! maybe can solve the ID moving and not in order problem
    #First FLATTEN the image, and then assign ID? then it should become in order
    #try try 
    img_shape = mask_ori_size.shape
    bb_df = leaf_kmeans_clustering(bounding_boxes, clust_by, number_of_classes)

    for c in range(number_of_classes):
        c_dirname = 'Class_' + str(c+1)
        if not os.path.exists(os.path.join(leaf_path, c_dirname)):
            os.mkdir(os.path.join(leaf_path,c_dirname))

    #DON'T USE LEAVES_IMG. or well can use geh... BUT save brightness_normalized to another folder
    if bright_norm != None:
        norm_dir = 'normalized_image'
        norm_path = os.path.join(leaf_path, norm_dir)
        if not os.path.exists(norm_path):
            os.mkdir(norm_path)
        bnorm_leaves = brightness_normalization(bright_norm, leaves_img, img_fname, norm_path, clip_hist_percent = 10)
        bn_dir = bright_norm
        for c in range(number_of_classes):
            c_dirname = 'Class_' + str(c+1)
            bn_path = os.path.join(leaf_path,c_dirname,bn_dir)
            if not os.path.exists(bn_path):
                os.mkdir(bn_path)
    elif bright_norm == None:
        pass

    #save each leaf photo into their class_dir
    for box in range(len(bb_df)):
        xmin = int(bb_df.iloc[box]['minx'] -10)
        xmax = int(bb_df.iloc[box]['maxx'] +10)
        ymin = int(bb_df.iloc[box]['miny'] -10)
        ymax = int(bb_df.iloc[box]['maxy'] +10)
        roi = leaves_img[ymin:ymax, xmin:xmax]
        leaf_id = str(bb_df.iloc[box]['leaf_ID'])
        leaf_class = str(int(bb_df.iloc[box]['clust_group']+1))
        leaf_fname = img_fname[:-4]+ '_LEAF_' + leaf_id + '.png'
        c_dir = 'Class_' + leaf_class
        cv2.imwrite(os.path.join(leaf_path, c_dir, leaf_fname), roi)
        if bright_norm != None:
            norm_roi = bnorm_leaves[ymin:ymax, xmin:xmax]
            bn_path = os.path.join(leaf_path, c_dir, bn_dir)
            cv2.imwrite(os.path.join(bn_path, leaf_fname), norm_roi)

    #image to check cluster grouping and give leaf ids
    cluster_grouping_check(img_path, bb_df, leaf_path, mask_ori_size)
    leaf_id_check(img_path, bb_df, leaf_path, mask_ori_size)


    #check for potential mask_overlaps
    c_df = bbox_measurements(bb_df)
    overlap_dir = 'potential_overlaps'
    if not os.path.exists(os.path.join(leaf_path,overlap_dir)):
        os.mkdir(os.path.join(leaf_path,overlap_dir))
    
    overlap_masks = 'masks'
    if not os.path.exists(os.path.join(leaf_path,overlap_dir,overlap_masks)):
        os.mkdir(os.path.join(leaf_path, overlap_dir, overlap_masks))
    
    c_masks = 'corrected_masks'
    if not os.path.exists(os.path.join(leaf_path,overlap_dir, c_masks)):
        os.mkdir(os.path.join(leaf_path, overlap_dir, c_masks))
    
    ms, sds = bbox_means_sds(c_df)

    msk_img = mask_ori_size
    #expand_dims 2 to add c to h,w,c format
    msk_img = np.expand_dims(msk_img,2)
    src_path = leaf_path
    dest_path = os.path.join(leaf_path, overlap_dir)
    msk_path = os.path.join(leaf_path,overlap_dir, overlap_masks)
    if len(overlap_ids) != 0:
        overlaps = overlap_ids
        if len(overlaps) != 0:
            potential_overlap_by_ids(overlaps,c_df,img_fname, src_path,dest_path,msk_path, msk_img)
        elif len(overlaps) == 0:
            potential_overlap_by_threshold(c_df, img_fname, ms, sds, src_path, dest_path, msk_path, msk_img)
    elif len(overlap_ids) == 0:
        potential_overlap_by_threshold(c_df, img_fname, ms, sds, src_path, dest_path, msk_path,msk_img)

    po_files = glob.glob(os.path.join(leaf_path, overlap_dir,'*.png'))
    '''
    if run_mode == 'check' and time_series_analysis == 'n':
        if len(po_files) != 0:
            print("The Leaf classifier inference has completed. mask corrections can be saved into the folder at this stage.\n")
            input("Press Enter to continue...\n")
    
    if run_mode == 'check' and time_series_analysis == 'y':
        if len(po_files) != 0 and t == 0:
            print("The Leaf classifier inference has completed. mask corrections can be saved into the folder at this stage.\n")
            input("Press Enter to continue...\n")
    '''
    if time_series_analysis == 'n':
        c_msk_files = glob.glob(os.path.join(leaf_path,overlap_dir, c_masks,'*.png'))
        if len(c_msk_files) != 0:
            cmask_proc(c_msk_files, c_df, leaf_path, overlap_dir, po_files,t,clust_by)

    if time_series_analysis == 'y' and t == 0:
        c_msk_files = glob.glob(os.path.join(leaf_path,overlap_dir, c_masks,'*.png'))
        leaf_path_0 = leaf_path
        if len(c_msk_files) != 0:
            cmask_proc(c_msk_files, c_df, leaf_path, overlap_dir, po_files,t, clust_by)

    elif time_series_analysis == 'y' and t != 0:
        #move c_msk_files from t=0 folder into t=t 
        t0_cmsks_abs_path = glob.glob(os.path.join(t0_fpath, overlap_dir, c_masks,'*.png'))
        t0_cmsks_fname = []
        for path in t0_cmsks_abs_path:
            f = os.path.split(path)[1]
            t0_cmsks_fname.append(f)
        for c in range(len(t0_cmsks_abs_path)):
            c_msk_dest = os.path.join(leaf_path, overlap_dir, c_masks,t0_cmsks_fname[c])
            c_msk_src = t0_cmsks_abs_path[c]
            shutil.copy(c_msk_src, c_msk_dest)
            #need to change the names of the masks so that it matches new folder

        c_msk_files = glob.glob(os.path.join(leaf_path,overlap_dir, c_masks,'*.png'))
        '''
        for f in range(len(c_msk_files)):
            lid = str(re.findall(r'\d+', c_msk_files[f])[-1])
            po_name = os.path.split(po_files[0])[1].rsplit('_',1)[0] + '_' + lid +'.png'
            c_msk_files[f] = po_name 
        '''
        if len(c_msk_files) != 0:
            cmask_proc(c_msk_files, c_df, leaf_path, overlap_dir, po_files,t,clust_by)



def Run_Lesion_Inference(leaf_res_path, number_of_classes, time_series_analysis, bright_norm, ts_res,t, tgi_params=[670, 550, 480], lesion_learner_path = 'Lesion_UNet/v1', size = 64,  stats_test = 'anova'):
    bn_dir = bright_norm
    leaf_ls = []
    leaf_NGRDI = []
    leaf_TGI = []
    leaf_lsn = []
    tgi_params = tgi_params
    for clas in range(number_of_classes):
        class_dir = 'Class_'+str(int(clas+1))
        #currently inference on bright norm leaves...
        fpaths = glob.glob(os.path.join(leaf_res_path,class_dir)+'/*.png')
        clas_ls = []
        clas_NGRDI = []
        clas_TGI = []
        clas_lsn = []
        rescale_dir = 'rescaled_leaves'
        rescale_path = os.path.join(leaf_res_path, class_dir, rescale_dir)
        if not os.path.exists(rescale_path):
            os.mkdir(rescale_path)
        for fp in fpaths:
            ori_leaf = cv2.imread(fp)
            h,w,c = ori_leaf.shape
            fname = os.path.split(fp)[1]
            img_rescale = image_rescale(fp, height = size, width = size, mode = "best")
            rescale_fpath = os.path.join(rescale_path, fname)
            cv2.imwrite(rescale_fpath, img_rescale)
            img = open_image(rescale_fpath)
            prediction_ls = Lesion_inference(lesion_learner_path, img)
            mask_np_ls = prediction_ls[1].numpy()
            mask_np_ls = np.uint8(mask_np_ls)
            mask_np_ls = image_rescale(mask_np_ls.squeeze(), height = h, width = w, mode = "best")
            mask_np_ls = np.expand_dims(mask_np_ls, 0)
            #save the mask image into a folder called Masks in the folder 
            mask_dir = os.path.join(leaf_res_path, class_dir, 'Masks')
            if not os.path.exists(mask_dir):
                os.mkdir(mask_dir)
            mask_fname = fname[:-4] + '_MASK.png'
            #remove pl_ccs at the end # # # # # # # # # # # 
            corrected_mask, leaf_label, lesion_label, img_state, label_image, leaf_labels = secondary_lesions_removal(mask_np_ls,mask_fname,mask_dir)
            #image_state error... seems like mask_gap_fill isn't working properly.
            #corrected_mask = mask_gap_fill(corrected_mask)
            img_state = len(np.unique(corrected_mask))
            imageio.imwrite(os.path.join(mask_dir,mask_fname), corrected_mask*127)
            #mask_check = skimage.measure.label(corrected_mask,background = 0, connectivity = 2)
            #check_ccs = len(np.unique(mask_check))
            if bright_norm == None:
                Lesion_size, TGI, NGRDI, lsn = leaf_measures_calculations(fp,None,corrected_mask, img_state, tgi_params, leaf_label, lesion_label)
            elif bright_norm != None:
                Lesion_size, TGI, NGRDI, lsn = leaf_measures_calculations(fp,bn_dir,corrected_mask, img_state, tgi_params, leaf_label, lesion_label)

            clas_ls.append(Lesion_size)
            clas_TGI.append(TGI)
            clas_NGRDI.append(NGRDI)
            clas_lsn.append(lsn)
            
        leaf_ls.append(clas_ls)
        leaf_TGI.append(clas_TGI)
        leaf_NGRDI.append(clas_NGRDI)
        leaf_lsn.append(clas_lsn)

    leaf_measures = [leaf_ls, leaf_TGI, leaf_NGRDI, leaf_lsn]
    #also save leaf_measures as a whole df so can see individual leaf data
    flat_lists = []
    for l in leaf_measures:
        flattened = [val for sublist in l for val in sublist]
        flat_lists.append(flattened)
    flat_arr = np.array(flat_lists)
    flat_df = pd.DataFrame(flat_lists, index = ["lesion_size(ratio)","TGI","NGRDI","lesion_size(pixels)"])
    flat_df = flat_df.T
    #write df to excel 
    flat_df_name = 'res.csv'
    flat_df_path = os.path.join(leaf_res_path, flat_df_name)
    flat_df.to_csv(flat_df_path)
    
    anova_p_vals, tukeys_res, group_means = stats_analysis(leaf_measures,stats_test)

    ls_mean = group_means[0]
    TGI_mean = group_means[1]
    NGRDI_mean = group_means[2]
    lsn_mean = group_means[3]
    
    ls_fname = fname[:-12] + '_lesion_sizes.txt'
    ls_fpath = os.path.join(leaf_res_path, ls_fname)
    TGI_fname = fname[:-12] + '_TGI.txt'
    TGI_fpath = os.path.join(leaf_res_path, TGI_fname)
    NGRDI_fname = fname[:-12] + '_NGRDI.txt'
    NGRDI_fpath = os.path.join(leaf_res_path, NGRDI_fname)


    #if len(ls_mean) == number_of_classes:
    ls_file = open(ls_fpath,'w')
    ls_file.write('lesion size (as ratio to leaf area) res: \n')
    for i in range(len(ls_mean)):
        ls_file.write('Group '+str(i+1)+' mean: ' + str(ls_mean[i])+'\n')
    ls_file.write('One Way Anova adjusted p value: ' + str(anova_p_vals[0])+'\n')
    ls_file.write('Tukey HSD res: ' + str(tukeys_res[0])+'\n')
    ls_file.write('lesion size (number of pixels): \n')
    for i in range(len(lsn_mean)):
        ls_file.write('Group ' + str(i+1) + ' mean: ' + str(lsn_mean[i])+'\n')
    ls_file.write('One Way Anova adjusted p value: ' + str(anova_p_vals[3])+'\n')
    ls_file.write('Tukey HSD res: ' + str(tukeys_res[3])+'\n')
    ls_file.close()
    
    #if len(TGI_mean) == number_of_classes:
    TGI_file = open(TGI_fpath,'w')
    TGI_file.write('TGI res: \n')
    for i in range(len(TGI_mean)):
        TGI_file.write('Group '+str(i+1)+' mean: ' + str(TGI_mean[i])+'\n')
    TGI_file.write('One Way Anova adjusted p value: ' + str(anova_p_vals[1])+'\n')
    TGI_file.write('Tukey HSD res: ' + str(tukeys_res[1])+'\n')
    TGI_file.close()
    
    #if len(NGRDI_mean) == number_of_classes:
    NGRDI_file = open(NGRDI_fpath,'w')
    NGRDI_file.write('NGRDI res: \n')
    for i in range(len(NGRDI_mean)):
        NGRDI_file.write('Group '+str(i+1)+' mean: ' + str(NGRDI_mean[i])+'\n')
    NGRDI_file.write('One Way Anova adjusted p value: ' + str(anova_p_vals[2])+'\n')
    NGRDI_file.write('Tukey HSD res: ' + str(tukeys_res[2])+'\n')
    NGRDI_file.close()
    
    if time_series_analysis == 'y':
        #append the group means to each df 
        for measure in range(len(group_means)):
            for group in range(number_of_classes):
                ts_res[group].iloc[measure,t] = group_means[measure][group]
    return ts_res
    

def compile_time_series_results(time_series_analysis, ts_res,out_dir, ts_dir, missing_tps, impute_method):
    #create_folder called time series analysis?
    ts_path = os.path.join(out_dir, ts_dir)
    if not os.path.exists(ts_path):
        os.mkdir(ts_path)

    #imputation for missing data if any is present
    if len(missing_tps) != 0:
        for m in range(4):
            measure_df = pd.DataFrame()
            for g in range(len(ts_res)):
                m_dat = ts_res[g].iloc[m]
                m_dat.name = 'Group_' + str(g+1)
                measure_df = measure_df.append(m_dat)

            if len(missing_tps) == 1:
                measure_df.insert(missing_tps[0],'missing',np.nan)
                new_end_time = end_time + 1
                measure_df.columns = range(start_time,new_end_time,interval)
            if len(missing_tps) > 1:
                for mp in range(len(missing_tps)):
                    if mp == 0:
                        measure_df.insert(missing_tps[mp],'missing',np.nan, allow_duplicates = True)
                    elif mp != 0 and missing_tps[mp] > missing_tps[mp-1]:
                        measure_df.insert(missing_tps[mp]+1,'missing',np.nan, allow_duplicates = True)
                    elif mp != 0 and missing_tps[mp] < missing_tps[mp-1]:
                        measure_df.insert(missing_tps[mp],'missing',np.nan, allow_duplicates = True)
                new_end_time = end_time + len(missing_tps)
                measure_df.columns = range(start_time, new_end_time, interval)
            #inserted new columns and renamed them... now impute based on selected method
            if impute_method in ['default predictive', 'least squares', 'stochastic', 'binary logistic', 'multinomial logistic', 'bayesian least squares', 'bayesian binary logistic', 'pmm', 'lrd', 'default univariate', 'default time', 'mean', 'median', 'mode', 'random', 'norm', 'categorical', 'interpolate', 'locf', 'nocb']:
                imputer = SingleImputer(strategy = impute_method, copy = True)
                t_df = measure_df.transpose()
                impute_df = imputer.fit_transform(t_df)
                impute_df = impute_df.transpose()
            else:
                impute_df = measure_df

            #shove rows of data back into ts_dfs
            for g in range(len(impute_df)):
                ts_res[g].iloc[m] = impute_df.iloc[g]

    '''
    for d in range(len(ts_res)):
        df_fname = 'Group_' + str(d+1) + '_ts.xlsx'
        df_fpath = os.path.join(ts_path, df_fname)
        ts_res[d].to_excel(df_fpath, sheet_name = 'Group_' + str(d+1))
    #take the ts_dfs and make graphs etc
    #make measure dfs as well
    '''

    m_names = ['lesion_size_ratio', 'TGI', 'NGRDI', 'lesion_size_pn']
    for measure in range(4):
        measure_df = pd.DataFrame()
        for group in range(len(ts_res)):
            m_dat = ts_res[group].iloc[measure]
            m_dat.name = 'Group_' + str(group+1)
            measure_df = measure_df.append(m_dat)
        mdf_fname = m_names[measure] + '.csv'
        mdf_fpath = os.path.join(ts_path, mdf_fname)
        measure_df.to_csv(mdf_fpath)

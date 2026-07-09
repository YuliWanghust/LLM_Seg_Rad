# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 09:38:15 2025

@author: leiy04
"""

import numpy as np
from convert import load_patient, struct_to_mask
from helper import get_subdir, get_lst, get_struct_name, mk_subdir, read_CT, write_dcm

pth_save = './GT'

lst = get_subdir('./data')

count = 0

expected_structs = {'GTV': ['GTV', 'GTVp'], 
                    'SmallBowel': ['SmallBowel'],
                    'LargeBowel': ['LargeBowel'],
                    'SpinalCord': ['SpinalCord'],
                    'Liver': ['Liver'],
                    'Duodenum': ['Duodenum'],
                    'Stomach': ['Stomach'],
                    'Kidney Lt': ['Kidney_L'],
                    'Kidney Rt': ['Kidney_R']}

for pth_i in lst:
    id_i = pth_i.split('/')[-1]
    print('Load Pt: ', id_i)
    
    dicom_files = get_lst(pth_i)
    patient = load_patient(dicom_files)
    
    lst_structure_name = get_struct_name(patient)
    
    matched_structs = {}
    missing_structs = []
    
    for std_name, aliases in expected_structs.items():
        found_match = False
        for s in lst_structure_name:
            for alias in aliases:
                if alias.lower() == s.lower():
                    matched_structs[std_name] = s  # record actual matched name
                    found_match = True
                    break
            if found_match:
                break
        if not found_match:
            missing_structs.append(std_name)

    if not missing_structs:
        print("All expected structures found.")
        
        save_dir = mk_subdir(pth_save, id_i)
        
        img, info = read_CT(pth_i)
        img_gt = np.zeros(img.shape) # save ground truth segmentation
        
        for ii, (k, v) in enumerate(matched_structs.items()):
            print(f" - {k} matched with '{v}'")
            
            gt = struct_to_mask(dicom_files, v)
            gt = np.transpose(gt, (1,2,0))
            img_gt[gt>0] = ii+1.0
            
        write_dcm(info, save_dir, img_gt)
        
        count += 1
    else:
        print("Missing structures (by alias check):", missing_structs)
    
print('# avilable Pt: ', count)
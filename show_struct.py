# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 08:02:22 2025

@author: leiy04
"""

from convert import load_patient
from helper import get_subdir, get_lst, get_struct_name

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
    print('Load Pt: ', pth_i.split('\\')[-1])
    
    dicom_files = get_lst(pth_i)
    patient = load_patient(dicom_files)
    
    lst_structure_name = get_struct_name(patient)
    print('Structures:', lst_structure_name)

    missing = []
    for std_name, aliases in expected_structs.items():
        if not any(any(alias.lower() == s.lower() for s in lst_structure_name) for alias in aliases):
            missing.append(std_name)

    if not missing:
        print("All expected structures found.")
        count += 1
    else:
        print("Missing structures (by alias check):", missing)
    
print('# avilable Pt: ', count)
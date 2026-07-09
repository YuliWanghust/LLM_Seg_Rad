# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 07:58:31 2025

@author: leiy04
"""

import os
import pydicom
import numpy as np
from pydicom.uid import generate_uid
from convert import load_patient, struct_to_mask

def get_subdir(pth):
    lst = []
    for dirName in os.listdir(pth):
        if os.path.isdir(os.path.join(pth,dirName)):
            lst.append(os.path.join(pth,dirName))
            lst.sort()
    return lst

def mk_subdir(pth_save,id_i):
    pth_save_sub = os.path.join(pth_save,id_i)
    if os.path.isdir(pth_save_sub)==False:
        os.mkdir(pth_save_sub)
    return pth_save_sub

def get_lst(pth):
    lst = []
    for root, _, files in os.walk(pth):
        for file in files:
            if file.upper() != 'DICOMDIR':
                lst.append(os.path.join(root, file))
    lst.sort()
    return lst

def get_tmp_lst(pth, mod='npy'):
    lst = []
    for root, _, files in os.walk(pth):
        for file in files:
            if (file.upper() != 'DICOMDIR') and (mod in file):
                lst.append(os.path.join(root, file))
    lst.sort()
    return lst

def get_intersection(lst1, lst2):
    lst1_name = [lst1[i].split('\\')[-1] for i in range(len(lst1))]
    lst2_name = [lst2[i].split('\\')[-1].split('.')[0].split('_')[1] for i in range(len(lst2))]
    both_name = list(set(lst1_name).intersection(set(lst2_name)))
    both_name.sort()
    
    idx_sub = [lst1_name.index(both_name[i]) for i in range(len(both_name))]
    lst1_sub = [lst1[i] for i in idx_sub]
    
    idx_sub = [lst2_name.index(both_name[i]) for i in range(len(both_name))]
    lst2_sub = [lst2[i] for i in idx_sub]
    return lst1_sub, lst2_sub

def get_mod_lst(pth, mod='RTSTRUCT'):
    lst = get_lst(pth)
    lst_sub = []
    for pth_dcm in lst:
        data = pydicom.dcmread(pth_dcm)
        if data.Modality == mod:
            lst_sub.append(pth_dcm)
    return lst_sub

def read_CT(pth):
    lst = get_mod_lst(pth, 'CT')
    # sort image index
    idx = [int(pydicom.dcmread(pth).InstanceNumber) for pth in lst]
    idx_sort = np.sort(np.array(idx))
    # read image from sorted index
    info = []
    img = []
    for j in range(len(idx_sort)):
        info.append(pydicom.dcmread(lst[idx.index(idx_sort[j])]))
        img.append(np.array(np.float32(info[j].pixel_array)*np.float32(info[j].RescaleSlope)+np.float32(info[j].RescaleIntercept)))
    img = np.stack(img,axis=2)
    return img, info

def write_dcm(info,path,NImg):
    Uid_study = generate_uid()
    Uid_series = generate_uid()
    Uid_for = generate_uid()
    for j in range(NImg.shape[2]):
        info[j].add_new(0x00281052, 'DS', NImg.min())
        info[j].add_new(0x00281053, 'DS', 1)
        temp = np.uint16((NImg[:,:,j]-np.float32(info[j].RescaleIntercept))/np.float32(info[j].RescaleSlope))
        info[j].PixelData = temp.tostring()
        info[j].add_new(0x0020000d, 'UI', Uid_study)
        info[j].add_new(0x0020000e, 'UI', Uid_series)
        info[j].add_new(0x00200052, 'UI', Uid_for)
        Uid_msop = generate_uid()
        Uid_sop = Uid_msop
        info[j].file_meta.add_new(0x00020003, 'UI', Uid_msop)
        info[j].add_new(0x00080018, 'UI', Uid_sop)
        # head, tail = os.path.split(lst_new[j])
        tail = str(j+1).zfill(4)+'.dcm'
        save_file = os.path.join(path,tail)
        pydicom.filewriter.dcmwrite(save_file, info[j], write_like_original=True)
        
# using libray of dicompylercore
def get_struct_name(patient):
    lst_key = list(patient['structures'].keys())
    return [patient['structures'][i]['name'] for i in lst_key]

def load_struct_name(pth):
    lst = get_lst(pth)
    patient = load_patient(lst)
    return get_struct_name(patient)

def get_binary_mask(pth, structure_name='Prostate'):
    lst = get_lst(pth)
    gt = struct_to_mask(lst, structure_name)
    gt = np.transpose(gt, (1,2,0))
    return gt

# used for central cropping
def get_crop_location(Img,threshold,box_size):
    mask = Img>threshold
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(mask)
    cen_idx = np.clip(int((rmin+rmax)/2),int(box_size[0]/2),Img.shape[0]-int(box_size[0]/2))
    cen_idy = np.clip(int((cmin+cmax)/2),int(box_size[1]/2),Img.shape[1]-int(box_size[1]/2))
    cen_idz = np.clip(int((zmin+zmax)/2),int(box_size[2]/2),Img.shape[2]-int(box_size[2]/2))
    return cen_idx-int(box_size[0]/2), cen_idx+int(box_size[0]/2), cen_idy-int(box_size[1]/2), cen_idy+int(box_size[1]/2), cen_idz-int(box_size[2]/2), cen_idz+int(box_size[2]/2)

def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax
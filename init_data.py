# -*- coding: utf-8 -*-
"""
Data Read-in Part
Read Data from ensemble-transfered data


09/2018 @ USTC ESS 1233
Fanghe : zfh1997 at mail.ustc.edu.cn(fzhao97 at gamil.com)
USTC-AEMOL
"""
import numpy as np
import netCDF4 as nc
import psutil as ps
import time
import os

def _sys_stat():
    """
    Print Memory Usage
    """
    mem_info = ps.virtual_memory()
    print("Memory Usage : " + str(mem_info.percent))

def graph_2d(variable):
    """
    This function is intentional to graph
    """
    fig, ax = plt.subplots()
    ax.imshow(variable)
    ax.set_title('CO')
    plt.show()
    return 0

def _message_display(string):
    """
    This function is only for display step message
    """
    print("========================================")
    print("            "+ string +"           ")
    print(time.asctime(time.localtime(time.time())))
    _sys_stat()
    print("========================================")

def read_wrf_data():
    """
    Read WRF meteorology data into the form:
    [batch_size, length, width, channel]

    Due to the bad mechanism of python-netcdf. This function would cost a lot of memory

        #Open File and Retrieve Data from it
        #Pay attention:
        #1. netCDF4 can not use with...as....: to open the file 
        #2. Do not close the file before u use the variables
        #3. Retrieve you data you exactly need

    To Fit the Network, data has been resize to 100 * 100

    Output data in form :

    Batch * lev * lat * lon * channel

    """
    emis_ensemble_file = r"D:\AEMOL\Tensorflow\Emission.nc"
    emis_label_file = r"D:\AEMOL\Tensorflow\Emission_label.nc"
    debug = False

    _message_display("Retrieve Data Start")

    #Retrieve Emission Label
    emis_label_fid = nc.Dataset(emis_label_file, "r")
    #Step 1. Retrieve and Resize
    dataset_key = list(emis_label_fid.variables.keys())
    emis_label = np.array(emis_label_fid.variables[dataset_key[0]])
    ##close file
    emis_label_fid.close()

    #Retrieve Emission Data
    emis_fid = nc.Dataset(emis_ensemble_file, "r")
    #Step 1. Retrieve and Resize
    dataset_key = list(emis_fid.variables.keys())
    emis_ensemble = np.array(emis_fid.variables[dataset_key[0]])
    ##close file
    emis_label_fid.close()

    _message_display("Retrieve Data End")

    print(emis_label.shape)
    print(emis_ensemble.shape)

    return emis_ensemble, wrf_out_data, emis_label


if __name__  == "__main__":
    read_wrf_data()

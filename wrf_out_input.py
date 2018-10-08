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
    print(string)
    print(time.asctime(time.localtime(time.time())))
    _sys_stat()
    print("========================================")

def decode(l):
    if isinstance(l, list):
        return [decode(x) for x in l]
    else:
        return l.decode()

def read_wrf_data():
    """
    Read WRF meteorology data into the form:
    [batch_size, length, width, channel]

    Due to the bad mechanism of python-netcdf. This function would cost a lot of memory

        #Open File and Retrieve Data from it
        #Pay attention:
        #1. Do not close the file before u use the variables
        #2. Retrieve you data you exactly need

    To Fit the Network, data has been resize to 100 * 100

    Output data in form :

    Batch * lev * lat * lon * channel

    """
    wrfout_ensemble_file = r"D:\AEMOL\Tensorflow\data_ensemble.nc"
    debug = False

    _message_display("Retrieve Data Start")

    #Retrieve WRFOUT Label
    with nc.Dataset(wrfout_ensemble_file, "r") as wrfout_fid:
        #Step 1. Retrieve and Resize
        dataset_key = list(wrfout_fid.variables.keys())

        # VAR 1 : TIME_TICK
        #Convert Times String from numpy byte to char
        wrf_timetick = list(wrfout_fid.variables['time'])

        shape = np.array(wrf_timetick).shape
        wrf_time = [([''] * shape[1]) for i in range(shape[0])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                 wrf_time[i][j] = wrf_timetick[i][j].decode()
        wrf_timetick = wrf_time

        wrf_time = [''.join(wrf_timetick[i]) for i in range(shape[0])]
        wrf_month = np.array(list(map(int, [(wrf_time[i])[5:7] for i in range(shape[0])] )))
        wrf_hour = np.array(list(map(int, [(wrf_time[i])[11:13] for i in range(shape[0])] )))

        # VAR 2 : Others
        u = np.array(wrfout_fid.variables['u'])[:, :, :, np.newaxis]
        v = np.array(wrfout_fid.variables['v'])[:, :, :, np.newaxis]
        w = np.array(wrfout_fid.variables['w'])[:, :, :, np.newaxis]
        T = np.array(wrfout_fid.variables['t'])[:, :, :, np.newaxis]
        co = np.array(wrfout_fid.variables['co'])[:, :, :, np.newaxis]
        rain = np.array(wrfout_fid.variables['rain'])[:, :, :, np.newaxis]

    #Concatenate all Channel into one array
    wrf_data = np.concatenate((u, v, w, T, rain, co), axis = 3)

    print(wrf_data.shape)

    _message_display("Retrieve Data End")
    return wrf_data, wrf_month, wrf_hour

def emis_cohere(wrf_month, wrf_hour):
    """
    Coherence Emission label with read in WRF data
    [batch_size, length, width, channel]

    To Fit the Network, data has been resize to 100 * 100

    Output data in form :

    Batch * lev * lat * lon * channel

    """
    emis_label_file = r"D:\AEMOL\Tensorflow\Emission_label.nc"
    debug = False
    data_size = wrf_month.shape[0]

    _message_display("Emission Coherence Start")

    #Retrieve WRFOUT Label
    with nc.Dataset(emis_label_file, "r") as emis_fid:

        # VAR Retrieve
        emis_label = np.array(emis_fid.variables['emission_label'])[:, 0, :, :, :]
        emis_label_output = emis_label
        emis_label = np.reshape(emis_label, (12, 24, 100, 100, 1))

        print(emis_label.shape)

    emis_output = np.zeros([data_size, 100, 100, 1])

    for i in range(data_size):
        emis_output[i, :, :, :] = emis_label[wrf_month[i] - 1, wrf_hour[i], :, :, :]

    _message_display("Emission Coherence End")

    return emis_output, emis_label_output


if __name__  == "__main__":
    wrf_data, wrf_month, wrf_hour = read_wrf_data()
    emis_output, emis_label_output = emis_cohere(wrf_month, wrf_hour)

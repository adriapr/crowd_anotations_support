# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:22:02 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import draw
from zipfile import ZipFile

import os

import load_data as crowdload

zip_path = os.path.join('data','tasks.zip')


def show_task(task_id, df_task):
    
    #Select corresponding task from data frame
    df = df_task.loc[df_task['task_id'] == task_id]
       
    subject_id = df['subject_id'][0]  #TODO also need to handle what happens if task_id doesn't exist
    airway_id =  df['airway_id'][0]
    
    
    task_file = 'data({}).airways({}).viewpoints(1).png'.format(subject_id, airway_id)
        
    #Unzip images if necessary
    with ZipFile(zip_path, 'r') as zip: 
        with zip.open(task_file) as myfile:
            im = mpimg.imread(myfile)
            plt.imshow(im, cmap="gray")
            plt.title('task {}, subject {}, airway {}'.format(task_id, subject_id, airway_id))
      


# Show all annotations of the same result
def show_result(result_id, df_task, df_res, df_annot):

    #Select corresponding result
    res = df_res.loc[df_res['result_id'] == result_id].reset_index()
         
    #Get result details and annotations
    task_id = res['task_id'][0]
    annot = df_annot.loc[df_annot['result_id'] == result_id]
    
    
    #Display everything
    ax = plt.gca()  
    
    show_task(task_id, df_task)
    

    for (index,a) in annot.iterrows():
    
        ell_patch, vertices = crowdload.get_ellipse_patch_vertices(a)
        ell_patch.set_edgecolor('#6699FF')
        ell_patch.set_linewidth(3)
        ax.add_patch(ell_patch)
        
    plt.xlabel('result {}'.format(result_id))
    plt.show()
      
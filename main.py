# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:47:40 2020

@author: vcheplyg
"""

import pandas as pd
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from skimage import draw
from parse import *

import load_data as crowdload


import analyze as crowdanalyze
import combine as crowdcombine

#Things that are working
def main():
    
    #crowdload.process_data()  #This needs to be redone if anything in the preprocessing changes! 
    
    # Load all the processed files 
    df_task, df_res, df_annot, df_truth, df_subject = crowdload.get_df_processed() 
    
    # Select valid results and analyze their statistics
    df_res_valid, df_res_invalid = crowdcombine.get_valid_results(df_res)
    
    # How many results are there? How many workers are there? ? 
    #crowdanalyze.print_result(df_res_valid, df_res_invalid)
    
    #crowdanalyze.print_worker(df_res)
    #crowdanalyze.plot_result_worker(df_res_valid)
    #crowdanalyze.scatter_worker_valid(df_res_valid, df_res_invalid)
    
    
    #Combine results per task in different ways, first, pick a random result
    #df_task_random = crowdcombine.get_task_random(df_task, df_res_valid)
    #crowdanalyze.scatter_correlation_expert_crowd(df_task_random, df_truth, 'random')
    
    
    #Combine all results per task with median combining
    df_task_median = crowdcombine.get_task_median(df_task, df_res_valid)
    #crowdanalyze.scatter_correlation_expert_crowd(df_task_median, df_truth, 'median')

    #Select best result per task (optimistically biased, uses ground truth!)
    #df_task_best = crowdcombine.get_task_best(df_task, df_res_valid, df_truth) 
    #crowdanalyze.scatter_correlation_expert_crowd(df_task_best, df_truth, 'best')


    #Correlation vs minimum number of available valid results 
    #crowdanalyze.plot_correlation_valid(df_task_random, df_truth, 'random')
    #crowdanalyze.plot_correlation_valid(df_task_median, df_truth, 'median')
    #crowdanalyze.plot_correlation_valid(df_task_best, df_truth, 'best')
    
    #crowdanalyze.plot_subject_correlation(df_subject, df_task_median, df_truth, 'median')

    #crowdanalyze.predict_subject_correlation(df_subject, df_task_median, df_truth, 'median')
    
    crowdanalyze.print_subject(df_subject, df_task_median, df_truth, 'median')
     
    #crowdanalyze.scatter_correlation_experts(df_task_median, df_truth, 'median')
     

#Development
def temp():
        
    
    #crowdload.process_data()  #This needs to be redone if anything in the preprocessing changes! 
        
    # Load all the processed files 
    df_task, df_res, df_annot, df_truth, df_subject = crowdload.get_df_processed() 
    
    # Select valid results and analyze their statistics
    df_res_valid, df_res_invalid = crowdcombine.get_valid_results(df_res)
    
    
    crowdanalyze.scatter_correlation_expert_crowd(df_res_valid, df_truth, '')
    
    #crowdanalyze.scatter_correlation_expert_crowd(df_res_valid, df_truth, '')
         
    # How many results are there? How many workers are there? ? 
    #crowdanalyze.print_result(df_res_valid, df_res_invalid)
    
    #df_task_random = crowdcombine.get_task_random(df_task, df_res_valid)
    #df_task_median = crowdcombine.get_task_median(df_task, df_res_valid)
    #df_task_best = crowdcombine.get_task_best(df_task, df_res_valid, df_truth)
   
    #crowdanalyze.predict_subject_correlation(df_subject, df_task_median, df_truth, 'median')

def debug_low():
    df_task, df_res, df_annot, df_truth, df_subject = crowdload.get_df_processed()
    df_res_valid, df_res_invalid = crowdcombine.get_valid_results(df_res)
    df1 = pd.merge(df_task, df_res_valid, on='subject_id', how='outer')
    df1 = pd.merge(df_task, df_res_valid, on='task_id', how='outer')
    
    #Subject with low WAP/WTR, but OK inner/outer
    df1 = df1.loc[df1['subject_id'] == 3]
    df1 = pd.merge(df_task, df_res_valid, on='task_id', how='outer')
    df2 = df1.loc[df1['subject_id'] == 3]
    
    df4 = pd.merge(df2, df_truth, on='task_id', how='outer')
    df4.plot.scatter('inner1','inner_median')
    df4.plot.scatter('inner1','inner')
    df4.plot.scatter('outer1','outer')
    df4.plot.scatter('wap1','wap')
    
    
    
    
    
    
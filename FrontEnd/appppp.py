# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:49:09 2023

@author: premp
"""
import os
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Parent directory:", parent_directory)




script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#        print("Parent directory:", parent_directory)
model_path = f'decision_tree_best_estimator.pkl'
#modelPKL_path = os.path.abspath(os.path.join(script_dir, 'models', model_path))
modelPKL_path = os.path.abspath(os.path.join(script_dir, 'BackEnd', model_path))

print(modelPKL_path)
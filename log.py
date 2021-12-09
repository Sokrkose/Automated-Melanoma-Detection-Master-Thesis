# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:53:43 2021

@author: Sokratis Koseoglou
"""

def log(model_name, acc_list, acc, epochs, batch_size, image_size):

    with open("results.txt", "a") as o:
        o.write("- Model: ")
        o.write(str(model_name))
        o.write(" has accuracy: ")
        o.write(str(acc_list))
        o.write(" with mean accuracy ")
        o.write(str(acc))
        o.write(" , ")
        o.write(str(epochs))
        o.write(" epochs.")
        o.write(" , ")
        o.write(str(batch_size))
        o.write(" batch_size.")
        o.write(" , ")
        o.write(str(image_size))
        o.write(" image_size.\n")
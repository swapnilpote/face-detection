import os, math, json, cv2, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# for FILENAME in *.jpg; do mv $FILENAME 2002_08_01_$FILENAME; done
orig_path = 'data/originalPics/'
path = 'data/'

print(os.getcwd())

for year in os.listdir(orig_path):
	print("year : " + year)
	for num in os.listdir(orig_path + year + '/'):
		print("num : " + num)
		for ano_num in os.listdir(orig_path + year + '/' + num + '/'):
			print("ano_num : " + ano_num)
			for FILENAME in os.listdir(orig_path + year + '/' + num + '/' + ano_num + '/big/'):
				print(FILENAME)

				PATH = os.getcwd() + '/' + orig_path + year + '/' + num + '/' + ano_num + '/big/'
				new_FILENAME = year + '_' + num + '_' + ano_num + '_' + FILENAME

				os.rename(PATH + FILENAME, os.getcwd() + '/' + path + 'train/face/' + new_FILENAME) 

import os 
path_to_data = '../../data/C50train/'

for subdir, dirs, files in os.walk(path_to_data):
    print(subdir)
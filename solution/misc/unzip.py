import numpy as np
import os
import zipfile

thumbnail_directory = '/data/suparna/workspace/TinyPortraits_thumbnails/'
zipfile_directory   = '/data/suparna/workspace/TinyPortraits/'

if not os.path.exists(thumbnail_directory):
    os.mkdir(thumbnail_directory)

# Load ordered list of archives
archive_file_list  = [file for file in sorted(os.listdir(zipfile_directory))
                      if file.split('.')[-1] == 'zip']
archive_file_count = len(archive_file_list)

# Check for completeness
assert archive_file_count == 66

for zip_archive_name in archive_file_list:
    
    # Verify archive name
    # zip_archive_name = 'Tiny_Portraits_Archive_{:03d}.zip'.format(index)
    # assert archive_file_list[index] == zip_archive_name
        
    # Extract image files 
    with zipfile.ZipFile(zipfile_directory + zip_archive_name, 'r') as archive:
        archive.extractall(thumbnail_directory)
            
    # Indicate progress
    num_images = len([file for file in os.listdir(thumbnail_directory) if file.split('.')[-1] == 'png'])
    #print('\rUnzipping archive #{:3d} ... Total images # {:6d}'.format(index, num_images), end = '')

# Verify number of images
assert num_images == 134734
            
print('\n*** DONE ***')
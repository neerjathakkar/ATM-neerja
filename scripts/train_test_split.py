import os
import random
import pickle

tracks_dir = "/home/neerja/tapnet-neerja/lion_dt_animal_320_points_10_seconds_256x256/tracks"

video_ids = []
total_files = 0
for file in os.listdir(tracks_dir):
    filename = os.path.basename(file).split('.')[0]
    if '_chunk_' in filename:
        # Find the position of '_chunk_' and extract everything before it
        chunk_pos = filename.find('_chunk_')
        video_id = filename[:chunk_pos]
    else:
        # Fallback to simple splitting if the expected pattern isn't found
        video_id = filename.split('_')[0]
    if video_id not in video_ids:
        video_ids.append(video_id)
    total_files += 1
print(video_ids)

# 80% train, 20% val
train_ids = random.sample(video_ids, int(0.8 * len(video_ids)))
val_ids = [id for id in video_ids if id not in train_ids]

print("train ids, in total", len(train_ids))
print(train_ids)
print("val ids, in total", len(val_ids))
print(val_ids)

# make train and val dirs  
root_dir = "/home/neerja/ATM-neerja/atm/lion_data"
os.makedirs(os.path.join(root_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(root_dir, "val"), exist_ok=True)

# move train and val files to train and val dirs
import shutil

val_files = 0
train_files = 0
for file in os.listdir(tracks_dir):
    if file.endswith('.pkl'):
        filename = os.path.basename(file).split('.')[0]
        if '_chunk_' in filename:
            # Extract the video ID from the filename
            chunk_pos = filename.find('_chunk_')
            video_id = filename[:chunk_pos]
            
            # Determine if this file belongs to train or val set
            if video_id in train_ids:
                shutil.copy(os.path.join(tracks_dir, file), os.path.join(root_dir, "train", file))
                train_files += 1
            elif video_id in val_ids:
                shutil.copy(os.path.join(tracks_dir, file), os.path.join(root_dir, "val", file))
                val_files += 1
print("train files, in total", train_files)
print("val files, in total", val_files)
print("total files, in total", total_files)
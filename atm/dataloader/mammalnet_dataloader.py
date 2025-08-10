import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from natsort import natsorted
import cv2
from einops import rearrange
import torchvision.transforms as transforms
   

import matplotlib.pyplot as plt
import mediapy as media
import colorsys
import random
from typing import List, Tuple, Optional


# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
  """Converts a sequence of points to color code video.

  Args:
    frames: [num_frames, height, width, 3], np.uint8, [0, 255]
    point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
    visibles: [num_points, num_frames], bool
    colormap: colormap for points, each point has a different RGB color.

  Returns:
    video: [num_frames, height, width, 3], np.uint8, [0, 255]
  """
  num_points, num_frames = point_tracks.shape[0:2]
  if colormap is None:
    colormap = get_colors(num_colors=num_points)
  height, width = frames.shape[1:3]
  dot_size_as_fraction_of_min_edge = 0.015
  radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
  diam = radius * 2 + 1
  quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
  quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
  icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
  sharpness = 0.15
  icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
  icon = 1 - icon[:, :, np.newaxis]
  icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
  icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
  icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
  icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

  video = frames.copy()
  for t in range(num_frames):
    # Pad so that points that extend outside the image frame don't crash us
    image = np.pad(
        video[t],
        [
            (radius + 1, radius + 1),
            (radius + 1, radius + 1),
            (0, 0),
        ],
    )
    for i in range(num_points):
      # The icon is centered at the center of a pixel, but the input coordinates
      # are raster coordinates.  Therefore, to render a point at (1,1) (which
      # lies on the corner between four pixels), we need 1/4 of the icon placed
      # centered on the 0'th row, 0'th column, etc.  We need to subtract
      # 0.5 to make the fractional position come out right.
      x, y = point_tracks[i, t, :] + 0.5
      x = min(max(x, 0.0), width)
      y = min(max(y, 0.0), height)

      if visibles[i, t]:
        x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        x2, y2 = x1 + 1, y1 + 1

        # bilinear interpolation
        patch = (
            icon1 * (x2 - x) * (y2 - y)
            + icon2 * (x2 - x) * (y - y1)
            + icon3 * (x - x1) * (y2 - y)
            + icon4 * (x - x1) * (y - y1)
        )
        x_ub = x1 + 2 * radius + 2
        y_ub = y1 + 2 * radius + 2
        image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
            y1:y_ub, x1:x_ub, :
        ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

      # Remove the pad
      video[t] = image[
          radius + 1 : -radius - 1, radius + 1 : -radius - 1
      ].astype(np.uint8)
  return video

class MammalNetDataset(Dataset):
    def __init__(self,
                 tracks_dir,           # Directory containing track PKL files
                 videos_dir,           # Directory containing video files
                 img_size=(128, 128),  # Target image size
                 num_track_ts=16,      # Number of timesteps for track sequences
                 num_track_ids=32,     # Number of track points to sample
                 frame_stack=1,        # Number of frames to stack
                 cache_all=False,      # Whether to cache all data in memory
                 cache_image=False,    # Whether to cache images in memory
                 num_demos=None,       # Percentage or number of demos to use
                 aug_prob=0.0,         # Probability of applying augmentation
                 video_extension='.mp4',  # Video file extension
                 task_embedding_dim=512): # Default task embedding dimension
        
        self.tracks_dir = tracks_dir
        self.videos_dir = videos_dir
        
        # Configuration parameters
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.frame_stack = frame_stack
        self.cache_all = cache_all
        self.cache_image = cache_image
        self.aug_prob = aug_prob
        self.video_extension = video_extension
        self.task_embedding_dim = task_embedding_dim
        
        # Initialize data structures for indexing
        self._cache = []
        self._index_to_demo_id = {}
        self._index_to_view_id = {}
        self._demo_id_to_path = {}
        self._demo_id_to_start_indices = {}
        self._demo_id_to_demo_length = {}
        self._index_to_animal_id = {}

        # Find all track files
        self.track_files = glob(os.path.join(tracks_dir, "*.pkl"))
        self.track_files = natsorted(self.track_files)
        
        if num_demos is not None:
            if isinstance(num_demos, float) and 0 < num_demos < 1:
                # Interpret as a percentage
                n_demo = int(len(self.track_files) * num_demos)
            else:
                # Interpret as a count
                n_demo = int(num_demos)
            self.track_files = self.track_files[:n_demo]
        
        print(f"Found {len(self.track_files)} track files in {self.tracks_dir}")
        
        # Set up augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        
        # Initialize dataset indices
        self._build_indices()
        
    def _build_indices(self):
        """Build indices for the dataset"""
        start_idx = 0
        
        for demo_idx, track_file in enumerate(self.track_files):
            # Extract video ID from filename
            # Extract video ID by removing the '_chunk_n_tracks' suffix
            filename = os.path.basename(track_file).split('.')[0]
            if '_chunk_' in filename:
                # Find the position of '_chunk_' and extract everything before it
                chunk_pos = filename.find('_chunk_')
                video_id = filename[:chunk_pos]
            else:
                # Fallback to simple splitting if the expected pattern isn't found
                video_id = filename.split('_')[0]
            video_path = os.path.join(self.videos_dir, f"{video_id}{self.video_extension}")
            # Load track data to get demo length
            with open(track_file, 'rb') as f:
                track_data = pickle.load(f)
            # Get demo length from track data
            global_shot = track_data['frame_range']
            demo_len = global_shot[1] - global_shot[0]
            
            # Skip demos that are too short
            if demo_len < self.num_track_ts:
                print(f"Skipping demo {demo_idx} with length {demo_len} (less than {self.num_track_ts} frames)")
                continue
                    
            effective_demo_len = max(0, demo_len - self.num_track_ts + 1)  # +1 since we can start at the last valid position
            
            # Process and cache demo if needed
            if self.cache_all:
                for animal_id in track_data['valid_animals']:
                    processed_demo = self._process_demo(track_data, video_path, animal_id, global_shot)
                    self._cache.append(processed_demo)
            
            # Dictionary to track valid segments per animal
            valid_segments = {}
            
            # Check each animal and identify valid segments
            for animal_id in track_data['valid_animals']:
                # import ipdb; ipdb.set_trace()

                within_mask = track_data['animal_tracks_within_seg'][animal_id]  # Shape (num_points, num_frames)
                valid_segments[animal_id] = []
                
                # Check each possible segment
                for segment_start in range(effective_demo_len):
                    segment_end = segment_start + self.num_track_ts
                    # Get visibility for this segment
                    segment_visibles = within_mask[:, segment_start:segment_end]  # Shape (num_points, num_track_ts)
                    # Count points visible in at least 1 frame in this segment
                    valid_points_count = np.sum(np.sum(segment_visibles, axis=1) >= 1)
                    
                    if valid_points_count >= self.num_track_ids:
                        valid_segments[animal_id].append(segment_start)
                        
            # Count total valid segments across all animals
            total_valid_segments = sum(len(segments) for segments in valid_segments.values())
            
            if total_valid_segments == 0:
                print(f"Skipping demo {demo_idx}: no valid segments found in any animal")
                continue
            
            # print(f"Processing demo {demo_idx} with {len(track_data['valid_animals'])} animals")
            # print(f"Demo length: {demo_len} frames, found {total_valid_segments} valid segments out of {effective_demo_len*len(track_data['valid_animals'])} possible segments")
            
            # Now build the indices based on valid segments
            for animal_id, segments in valid_segments.items():
                if not segments:  # Skip animals with no valid segments
                    print(f"Animal {animal_id} in demo {demo_idx} has no valid segments, skipping")
                    continue
                    
                # print(f"Animal {animal_id} in demo {demo_idx} has {len(segments)} valid segments")
                
                # Store the starting index for this demo and animal
                if demo_idx not in self._demo_id_to_start_indices:
                    self._demo_id_to_start_indices[demo_idx] = {}
                
                self._demo_id_to_start_indices[demo_idx][animal_id] = start_idx
                
                # Map indices to demo, view, and animal
                for segment_start in segments:
                    self._index_to_demo_id[start_idx] = demo_idx
                    self._index_to_view_id[start_idx] = segment_start
                    self._index_to_animal_id[start_idx] = animal_id
                    start_idx += 1
                # print(f"Stored start index {start_idx} for demo {demo_idx}, animal {animal_id}")
            
            # Store paths and lengths
            self._demo_id_to_path[demo_idx] = track_file
            self._demo_id_to_demo_length[demo_idx] = demo_len
            
            # print(f"Updated start_idx to {start_idx}")
        
        self.total_samples = start_idx
        # print(f"Total samples after filtering: {self.total_samples}")
    def _build_indices_old(self):
        """Build indices for the dataset"""
        start_idx = 0
        
        for demo_idx, track_file in enumerate(self.track_files):
            # Extract video ID from filename
            video_id = os.path.basename(track_file).split('.')[0].split('_')[0]
            video_path = os.path.join(self.videos_dir, f"{video_id}{self.video_extension}")
            # Load track data to get demo length
            with open(track_file, 'rb') as f:
                track_data = pickle.load(f)
            # Get demo length from track data
            # Assuming tracks are stored with shape (frames, num_points, 2)
            global_shot = track_data['frame_range']
            demo_len = global_shot[1] - global_shot[0]
            
            # Skip demos that are too short
            if demo_len < self.num_track_ts:
                print(f"Skipping demo {demo_idx} with length {demo_len} (less than {self.num_track_ts} frames)")
                continue
                
            effective_demo_len = max(0, demo_len - self.num_track_ts)

            num_valid_animals = len(track_data['valid_animals'])
            
            # Process and cache demo if needed
            if self.cache_all:
                for animal_id in track_data['valid_animals']:
                    processed_demo = self._process_demo(track_data, video_path, animal_id, global_shot)
                    self._cache.append(processed_demo)
            
            # Store indices for each animal in this demo
            # print(f"Processing demo {demo_idx} with {num_valid_animals} valid animals")
            # print(f"Demo length: {demo_len} frames")
            
            for animal_idx, animal_id in enumerate(track_data['valid_animals']):
                animal_start_idx = start_idx + animal_idx * effective_demo_len 
                animal_end_idx = animal_start_idx + effective_demo_len 
                
                # print(f"Animal {animal_id} (idx {animal_idx}): indices {animal_start_idx} to {animal_end_idx}")
                
                # Map indices to demo and animal
                for k in range(animal_start_idx, animal_end_idx):
                    self._index_to_demo_id[k] = demo_idx
                    self._index_to_view_id[k] = (k - animal_start_idx)  
                    self._index_to_animal_id[k] = animal_id
                
                # Store the starting index for this demo and animal
                if demo_idx not in self._demo_id_to_start_indices:
                    self._demo_id_to_start_indices[demo_idx] = {}
                self._demo_id_to_start_indices[demo_idx][animal_id] = animal_start_idx
                
                # print(f"Stored start index {animal_start_idx} for demo {demo_idx}, animal {animal_id}")
            
            # Store paths and lengths
            self._demo_id_to_path[demo_idx] = track_file
            self._demo_id_to_demo_length[demo_idx] = demo_len
            
            # print(f"Updated demo {demo_idx} path and length")
            
            # Update the global index counter
            start_idx += effective_demo_len * num_valid_animals
            # print(f"New start_idx: {start_idx}")
            
        self.total_samples = start_idx
        # print(f"Total samples: {self.total_samples}")
    
    def _process_demo(self, track_data, video_path, animal_id, global_shot):
        """Process demo data for caching"""
        processed_demo = {
            'video_path': video_path,
            'tracks': torch.from_numpy(track_data['animal_tracks'][animal_id]).float(),  # (num_points, frames, 2)
            'visibility': torch.from_numpy(track_data['animal_visibles'][animal_id]).float(),  # (num_points, frames)
            'task_emb': None
            # 'task_emb': torch.from_numpy(
            #     track_data.get('task_emb', np.zeros(self.task_embedding_dim))
            # ).float()  # (dim,)
        }
        
        if self.cache_image:
            # Load all video frames
            frames = self._load_all_video_frames(video_path, global_shot)
            processed_demo['video_frames'] = frames
        
        return processed_demo
    
    def _load_all_video_frames(self, video_path, global_shot):
        """Load all frames from a video file between global_shot start and end frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        # Get original width and height
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check if video opened successfully
        if not cap.isOpened():
            import ipdb; ipdb.set_trace()
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Seek to start frame
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, global_shot[0]):
            raise ValueError(f"Failed to seek to frame {global_shot[0]} in video: {video_path}")
        
        # Read frames until end frame
        frame_count = 0
        expected_frames = global_shot[1] - global_shot[0]
        
        for _ in range(expected_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could only read {frame_count}/{expected_frames} frames from {video_path}")
                break
                
            try:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if frame.shape[0] != self.img_size[0] or frame.shape[1] != self.img_size[1]:
                    frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
                
                frames.append(frame)
                frame_count += 1
            except Exception as e:
                print(f"Error processing frame {global_shot[0] + frame_count}: {str(e)}")
                continue
        
        if frame_count == 0:
            raise ValueError(f"Failed to read any frames from {video_path} between {global_shot[0]} and {global_shot[1]}")
        
        if frame_count < expected_frames:
            print(f"Warning: Expected {expected_frames} frames but got {frame_count} from {video_path}")
        # print(f"got {len(frames)} frames")
        
        cap.release()
        # Convert frames to tensor with shape (t, c, h, w)
        frames = np.stack(frames)  # (t, h, w, c)
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = torch.from_numpy(frames).float()
        
        return frames, (orig_width, orig_height)

        
    def _load_selected_video_frames(self, video_path, start_idx, end_idx):
        """Load all frames from a video file between start_idx and end_idx"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        # Get original width and height
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check if video opened successfully
        if not cap.isOpened():
            import ipdb; ipdb.set_trace()
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Seek to start frame
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx):
            raise ValueError(f"Failed to seek to frame {start_idx} in video: {video_path}")
        
        # Read frames until end frame
        frame_count = 0
        expected_frames = end_idx - start_idx
        
        for _ in range(expected_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could only read {frame_count}/{expected_frames} frames from {video_path}")
                break
                
            try:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if frame.shape[0] != self.img_size[0] or frame.shape[1] != self.img_size[1]:
                    frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
                
                frames.append(frame)
                frame_count += 1
            except Exception as e:
                print(f"Error processing frame {start_idx + frame_count}: {str(e)}")
                continue
        
        if frame_count == 0:
            raise ValueError(f"Failed to read any frames from {video_path} between {start_idx} and {end_idx}")
        
        if frame_count < expected_frames:
            print(f"Warning: Expected {expected_frames} frames but got {frame_count} from {video_path}")
        # print(f"got {len(frames)} frames")
        
        cap.release()
        # Convert frames to tensor with shape (t, c, h, w)
        frames = np.stack(frames)  # (t, h, w, c)
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = torch.from_numpy(frames).float()
        
        return frames, (orig_width, orig_height)
  
    # TODO this does not work
    def _sample_tracks_random(self, tracks, visibility, num_timesteps, num_points):
        # import ipdb; ipdb.set_trace()
        # tracks: [num_points, num_frames, 2]
        # visibility: [num_points, num_frames]
        if isinstance(tracks, np.ndarray):
            tracks = torch.from_numpy(tracks).float()
        if isinstance(visibility, np.ndarray):
            visibility = torch.from_numpy(visibility).float()
        
        # Sample in time - Ensure we have enough points to sample a contiguous chunk
        max_start_idx = tracks.shape[1] - num_timesteps
        # if max_start_idx < 0:
        #     # Not enough points, so we'll just use all available and repeat the last one
        #     start_idx = 0
        #     end_idx = tracks.shape[1]
        #     selected_indices_timesteps = torch.arange(start_idx, end_idx)
        #     # Pad with repetitions of the last index if needed
        #     if len(selected_indices_timesteps) < num_timesteps:
        #         padding = torch.full((num_timesteps - len(selected_indices_timesteps),), end_idx - 1, 
        #                             dtype=torch.long, device=selected_indices_timesteps.device)
        #         selected_indices_timesteps = torch.cat([selected_indices_timesteps, padding])
        # else:
        # Randomly select a starting point and take num_samples consecutive indices
        start_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
        # print("selected start idx", start_idx)
        end_idx = start_idx + num_timesteps
        # print("end idx", end_idx)
        selected_indices_timesteps = torch.arange(start_idx, end_idx)
        
        sampled_tracks_timesteps = tracks[:, selected_indices_timesteps]
        sampled_visibility_timesteps = visibility[:, selected_indices_timesteps]
        # print("sampled tracks timesteps shape", sampled_tracks_timesteps.shape)
        # sample num_points points
        selected_indices = torch.randint(0, tracks.shape[0], (num_points,))
        sampled_tracks = sampled_tracks_timesteps[selected_indices]
        sampled_visibility = sampled_visibility_timesteps[selected_indices]
        # print("sampled tracks shape", sampled_tracks.shape)
        return sampled_tracks, sampled_visibility, selected_indices_timesteps
    
    def _sample_tracks(self, tracks, visibility, num_samples):
        """
        Sample tracks prioritizing visible points first
        
        Args:
            tracks: tensor of shape (track_len, n, 2)
            visibility: tensor of shape (track_len, n)
            num_samples: number of track points to sample
        
        Returns:
            sampled_tracks: tensor of shape (track_len, num_samples, 2)
            sampled_visibility: tensor of shape (track_len, num_samples)
        """
        if isinstance(tracks, np.ndarray):
            tracks = torch.from_numpy(tracks).float()
        if isinstance(visibility, np.ndarray):
            visibility = torch.from_numpy(visibility).float()
        
        # Get the average visibility per track point
        mean_visibility = visibility.mean(dim=0)  # (n,)
        
        # Sort track points by visibility
        sorted_indices = torch.argsort(mean_visibility, descending=True)
        
        # Take top num_samples points
        selected_indices = sorted_indices[:num_samples]
        
        # Pad with zeros if we don't have enough points
        if len(selected_indices) < num_samples:
            pad_size = num_samples - len(selected_indices)
            if len(selected_indices) > 0:
                # Repeat the last index
                padding = torch.full((pad_size,), selected_indices[-1], device=selected_indices.device, dtype=selected_indices.dtype)
            else:
                # Create dummy indices
                padding = torch.zeros(pad_size, device=tracks.device, dtype=torch.long)
            selected_indices = torch.cat([selected_indices, padding])
        
        # Sample tracks and visibility
        sampled_tracks = tracks[:, selected_indices]
        sampled_visibility = visibility[:, selected_indices]
        
        return sampled_tracks, sampled_visibility
    
    def _augment_data(self, video, tracks):
        """Apply data augmentation to video and tracks"""
        # Convert to PIL Image format for torchvision transforms
        b, t, c, h, w = video.shape
        
        # Apply color jitter to video
        augmented_video = torch.zeros_like(video)
        for b_idx in range(b):
            for t_idx in range(t):
                frame = video[b_idx, t_idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                frame = transforms.ToPILImage()(frame)
                frame = self.color_jitter(frame)
                frame = transforms.ToTensor()(frame) * 255.0
                augmented_video[b_idx, t_idx] = frame
        
        # We're not modifying the tracks for this simple augmentation
        # For more complex augmentations like shifts or rotations, 
        # you would need to update the track coordinates correspondingly
        
        return augmented_video, tracks
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.total_samples
    
    def __getitem__(self, index):
        """Get a sample from the dataset"""
        demo_id = self._index_to_demo_id[index]
        view_id = self._index_to_view_id[index]
        # print("dataloader index", index)
        # print("view id", view_id)
        animal_id = self._index_to_animal_id[index]
        animal_start_index = self._demo_id_to_start_indices[demo_id][animal_id]
        path = self._demo_id_to_path[demo_id]
        # print("path", path)
        # import ipdb; ipdb.set_trace()
        # Calculate the time offset within the demo
        # time_offset = (index - animal_start_index) // 2
        time_offset = view_id
        
        # TODO - implement/fix this
        if self.cache_all:
            demo = self._cache[demo_id]
            
            if self.cache_image:
                # Get frames from cached video
                start_idx = max(0, time_offset - self.frame_stack + 1)
                end_idx = time_offset + 1
                vids = demo['video_frames'][start_idx:end_idx]
                
                # Pad if needed
                if len(vids) < self.frame_stack:
                    if len(vids) > 0:
                        # Pad with first frame
                        pad = torch.tile(vids[0:1], (self.frame_stack - len(vids), 1, 1, 1))
                        vids = torch.cat([pad, vids], dim=0)
                    else:
                        # Create empty frames
                        vids = torch.zeros((self.frame_stack, 3, self.img_size[0], self.img_size[1]), dtype=torch.float32)
            else:
                # Load frames from video file
                vids = self._load_video_frames(
                    demo['video_path'],
                    time_offset,
                    self.frame_stack,
                    backward=True
                )
            
            # Get track data
            end_time = min(time_offset + self.num_track_ts, len(demo['tracks']))
            tracks = demo['tracks'][time_offset:end_time]
            visibility = demo['visibility'][time_offset:end_time]
            
            # Pad tracks and visibility if needed
            if len(tracks) < self.num_track_ts:
                pad_tracks = torch.tile(tracks[-1:], (self.num_track_ts - len(tracks), 1, 1))
                pad_vis = torch.tile(visibility[-1:], (self.num_track_ts - len(visibility), 1))
                tracks = torch.cat([tracks, pad_tracks], dim=0)
                visibility = torch.cat([visibility, pad_vis], dim=0)
            
            task_emb = demo['task_emb']
            
        else:
            # Load data directly from files
            track_file = self._demo_id_to_path[demo_id]
            
            # Extract video ID from filename
            filename = os.path.basename(track_file).split('.')[0]
            if '_chunk_' in filename:
                # Find the position of '_chunk_' and extract everything before it
                chunk_pos = filename.find('_chunk_')
                video_id = filename[:chunk_pos]
            else:
                # Fallback to simple splitting if the expected pattern isn't found
                video_id = filename.split('_')[0]
            video_path = os.path.join(self.videos_dir, f"{video_id}{self.video_extension}") 
            
            # Load track data
            with open(track_file, 'rb') as f:
                track_data = pickle.load(f)

            global_shot = track_data['frame_range']
            start_frame = view_id + global_shot[0]
            end_frame = start_frame + self.num_track_ts
            # Load video frames
            vids, (orig_width, orig_height) = self._load_selected_video_frames(video_path, start_frame, end_frame)
            
            # Get track information
            tracks = track_data['animal_tracks'][animal_id]
            visibility = track_data['animal_visibles'][animal_id]
            within_seg = track_data['animal_tracks_within_seg'][animal_id]

            # normalize tracks
            tracks[:, :, 0] = tracks[:, :, 0] / orig_width
            tracks[:, :, 1] = tracks[:, :, 1] / orig_height
        
            # # Pad tracks and visibility if needed
            # if len(tracks) < self.num_track_ts:
            #     if isinstance(tracks, np.ndarray):
            #         pad_tracks = np.tile(tracks[-1:], (self.num_track_ts - len(tracks), 1, 1))
            #         pad_vis = np.tile(visibility[-1:], (self.num_track_ts - len(visibility), 1))
            #         tracks = np.concatenate([tracks, pad_tracks], axis=0)
            #         visibility = np.concatenate([visibility, pad_vis], axis=0)
            #     else:
            #         pad_tracks = torch.tile(tracks[-1:], (self.num_track_ts - len(tracks), 1, 1))
            #         pad_vis = torch.tile(visibility[-1:], (self.num_track_ts - len(visibility), 1))
            #         tracks = torch.cat([tracks, pad_tracks], dim=0)
            #         visibility = torch.cat([visibility, pad_vis], dim=0)
            
            # Get task embedding
            # task_emb = track_data.get('task_emb', np.zeros(self.task_embedding_dim))
            task_emb = torch.zeros(self.task_embedding_dim)
        
        # Apply augmentation if needed
        # TODO - implement/fix this
        # if np.random.rand() < self.aug_prob:
        #     vids = vids.unsqueeze(0)  # Add batch dimension (1, t, c, h, w)
        #     tracks = tracks.unsqueeze(0).unsqueeze(0)  # Add batch dimension (1, 1, track_len, n, 2)
        #     vids, tracks = self._augment_data(vids, tracks)
        #     vids = vids[0]  # Remove batch dimension
        #     tracks = tracks[0, 0]  # Remove batch dimension
        
        
        # Sample tracks to get the required number of track points
        # tracks, visibility, selected_indices = self._sample_tracks_random(tracks, visibility, self.num_track_ts, self.num_track_ids)
        # import ipdb; ipdb.set_trace()
        tracks = tracks[:self.num_track_ids, view_id:view_id+self.num_track_ts, :]
        # note: this g
        # tracks = tracks[:self.num_track_ids, :self.num_track_ts, :]
        # tracks = rearrange(tracks, 'n t 2 -> t n 2')
        tracks = np.transpose(tracks, (1, 0, 2))
        visibility = visibility[:self.num_track_ids, :self.num_track_ts]
        visibility = np.transpose(visibility, (1, 0))
        within_seg = within_seg[:self.num_track_ids, :self.num_track_ts]
        # selected_indices = torch.arange(global_shot[0], global_shot[0] + self.num_track_ts)
        
        return {
            'video': vids,
            'tracks': tracks, 
            'visibility': visibility,
            'task_emb': task_emb,
            'track_file': track_file,
            'animal_id': animal_id,
            'shot_range': global_shot,
            'frame_start': view_id, # TODO should this be view_id + global_shot[0]?
            'within_seg': within_seg
        }

def test_pkls():
    video_dir = "/datasets/mammal_net/current/full_videos/"
    track_dir = "/home/neerja/tapnet-neerja/lion_dt_animal_320_points_10_seconds_256x256/tracks"
    for file in os.listdir(track_dir):
        file = "0L7-8yqgMEo_chunk_0_tracks.pkl"
        if file.endswith(".pkl"):
            print(file)
            track_file = os.path.join(track_dir, file)
            with open(track_file, 'rb') as f:
                track_data = pickle.load(f)
            print(track_data.keys())
        filename = os.path.basename(track_file).split('.')[0]
        if '_chunk_' in filename:
            # Find the position of '_chunk_' and extract everything before it
            chunk_pos = filename.find('_chunk_')
            video_id = filename[:chunk_pos]
        else:
            # Fallback to simple splitting if the expected pattern isn't found
            video_id = filename.split('_')[0]
        video_path = os.path.join(video_dir, f"{video_id}.mp4") 
        print("reading video")
        video = media.read_video(video_path)
        print("read video")
        for animal_id in track_data['valid_animals']:
            print("painting track for animal", animal_id)
            tracks = track_data['animal_tracks'][animal_id]
            visibility = track_data['animal_visibles'][animal_id]
            print("tracks shape", tracks.shape)
            print("visibility shape", visibility.shape)
            video_viz = paint_point_track(video, tracks, visibility)
            print("painting done")
            media.write_video(f"track_visualization/{video_id}_sanity_check.mp4", video_viz, fps=30, codec='libx264')
            print("writing video to sanity check")
            break
        print("done")
    
if __name__ == "__main__":
    # test_pkls()
    video_dir = "/datasets/mammal_net/current/full_videos/"
    dataset = MammalNetDataset(tracks_dir="/home/neerja/tapnet-neerja/lion_dt_animal_320_points_10_seconds_256x256/tracks", 
                               videos_dir=video_dir, 
                               img_size=(128, 128), 
                               num_track_ts=16, 
                               num_track_ids=32, 
                               frame_stack=1, 
                               cache_all=False, 
                               cache_image=False, 
                               num_demos=0.1, 
                               aug_prob=0.5)
    print(len(dataset))
    
    # print(dataset[0])
    def visualize_tracks(dataset, index, video_dir, output_dir="track_visualization"):
        """
        Visualize tracks from a dataset sample
        
        Args:
            dataset: The MammalNetDataset instance
            index: Index of the sample to visualize
            video_dir: Directory containing the videos
            output_dir: Directory to save the visualization
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sample data
        sample = dataset[index]
        video = sample['video']
        tracks = sample['tracks']
        visibility = sample['visibility']
        task_emb = sample['task_emb']
        shot_range = sample['shot_range']
        track_file = sample['track_file']
        frame_start = sample['frame_start']
        # selected_indices = sample['selected_indices']
        num_track_ts = 16 

        print(track_file)
        print("frame start", frame_start)
        print("shot range", shot_range)
        print("tracks shape", tracks.shape)
        print("visibility shape", visibility.shape)

        # Read the original video
        print("reading video")
        filename = os.path.basename(track_file).split('.')[0]
        if '_chunk_' in filename:
            # Find the position of '_chunk_' and extract everything before it
            chunk_pos = filename.find('_chunk_')
            video_id = filename[:chunk_pos]
        else:
            # Fallback to simple splitting if the expected pattern isn't found
            video_id = filename.split('_')[0]
        video_path = os.path.join(video_dir, f"{video_id}.mp4") 
        video = media.read_video(video_path)
        
        # Extract the relevant portion of the video
        global_start = shot_range[0] + frame_start
        global_end = global_start + num_track_ts
        video = video[global_start:global_end]
        print(f"Using frames from {global_start} to {global_end}")
        
        # Get video dimensions
        video_width = video.shape[2]
        video_height = video.shape[1]

        print("unnormalizing tracks")
        
        # Unnormalize tracks to pixel coordinates
        tracks[:, :, 0] = tracks[:, :, 0] * video_width
        tracks[:, :, 1] = tracks[:, :, 1] * video_height
        
        # Convert to numpy if needed
        if not isinstance(tracks, np.ndarray):
            tracks = tracks.numpy()
        if not isinstance(visibility, np.ndarray):
            visibility = visibility.numpy()
            
        # Transpose to match expected format for paint_point_track
        tracks = np.transpose(tracks, (1, 0, 2))
        visibility = np.transpose(visibility, (1, 0))

        # Paint tracks on video and save
        video = paint_point_track(video, tracks, visibility)
        output_path = f"{output_dir}/{video_id}_index_{index}_rearranged_tracks_sampled_start_{frame_start}.mp4"
        media.write_video(output_path, video, fps=30, codec='libx264')
        
        return output_path

    # for i in [0, 143, 900, 3092, 1143]:
    #     visualize_tracks(dataset, i, video_dir)
    for i in range(0, 74000, 1000):
        visualize_tracks(dataset, i, video_dir)
    
    # Example usage
    # visualize_tracks(dataset, 38000, video_dir)
  
    

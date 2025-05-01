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
        num_views = 1  # 1 per frame
        
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

            num_valid_animals = len(track_data['valid_animals'])
            
            # Process and cache demo if needed
            if self.cache_all:
                for animal_id in track_data['valid_animals']:
                    processed_demo = self._process_demo(track_data, video_path, animal_id, global_shot)
                    self._cache.append(processed_demo)
            
            # Store indices for each animal in this demo
            print(f"Processing demo {demo_idx} with {num_valid_animals} valid animals")
            print(f"Demo length: {demo_len} frames")
            
            for animal_idx, animal_id in enumerate(track_data['valid_animals']):
                animal_start_idx = start_idx + animal_idx * demo_len * num_views
                animal_end_idx = animal_start_idx + demo_len * num_views
                
                print(f"Animal {animal_id} (idx {animal_idx}): indices {animal_start_idx} to {animal_end_idx}")
                
                # Map indices to demo and animal
                for k in range(animal_start_idx, animal_end_idx):
                    self._index_to_demo_id[k] = demo_idx
                    self._index_to_view_id[k] = (k - animal_start_idx) % num_views
                    self._index_to_animal_id[k] = animal_id
                
                # Store the starting index for this demo and animal
                if demo_idx not in self._demo_id_to_start_indices:
                    self._demo_id_to_start_indices[demo_idx] = {}
                self._demo_id_to_start_indices[demo_idx][animal_id] = animal_start_idx
                
                print(f"Stored start index {animal_start_idx} for demo {demo_idx}, animal {animal_id}")
            
            # Store paths and lengths
            self._demo_id_to_path[demo_idx] = track_file
            self._demo_id_to_demo_length[demo_idx] = demo_len
            
            print(f"Updated demo {demo_idx} path and length")
            
            # Update the global index counter
            start_idx += demo_len * num_views * num_valid_animals
            print(f"New start_idx: {start_idx}")
            
        self.total_samples = start_idx
        print(f"Total samples: {self.total_samples}")
    
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
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_shot[0])
        
        # Read frames until end frame
        for _ in range(global_shot[1] - global_shot[0]):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if frame.shape[0] != self.img_size[0] or frame.shape[1] != self.img_size[1]:
                frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
        
        cap.release()
        # Convert frames to tensor with shape (t, c, h, w)
        frames = np.stack(frames)  # (t, h, w, c)
        frames = rearrange(frames, "t h w c -> t c h w")
        frames = torch.from_numpy(frames).float()
        
        return frames
    
  
    def _sample_tracks_random(self, tracks, visibility, num_samples):
        selected_indices = torch.randint(0, tracks.shape[1], (num_samples,))
        sampled_tracks = tracks[:, selected_indices]
        sampled_visibility = visibility[:, selected_indices]
        
        return sampled_tracks, sampled_visibility# Get track information for the specific animal
    
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
        animal_id = self._index_to_animal_id[index]
        animal_start_index = self._demo_id_to_start_indices[demo_id][animal_id]
        
        # Calculate the time offset within the demo
        time_offset = (index - animal_start_index) // 2
        
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
            # import ipdb; ipdb.set_trace()
            # Load data directly from files
            track_file = self._demo_id_to_path[demo_id]
            
            # Extract video ID from filename
            video_id = os.path.basename(track_file).split('.')[0].split('_')[0]
            video_path = os.path.join(self.videos_dir, f"{video_id}{self.video_extension}") 
            
            # Load track data
            with open(track_file, 'rb') as f:
                track_data = pickle.load(f)

            global_shot = track_data['frame_range']
            # Load video frames
            vids = self._load_all_video_frames(video_path, global_shot)
            
            # Get track information
            tracks = track_data['animal_tracks'][animal_id][time_offset:time_offset + self.num_track_ts]
            visibility = track_data['animal_visibles'][animal_id][time_offset:time_offset + self.num_track_ts]
        
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
            task_emb = None
        
        # Apply augmentation if needed
        # TODO - implement/fix this
        # if np.random.rand() < self.aug_prob:
        #     vids = vids.unsqueeze(0)  # Add batch dimension (1, t, c, h, w)
        #     tracks = tracks.unsqueeze(0).unsqueeze(0)  # Add batch dimension (1, 1, track_len, n, 2)
        #     vids, tracks = self._augment_data(vids, tracks)
        #     vids = vids[0]  # Remove batch dimension
        #     tracks = tracks[0, 0]  # Remove batch dimension
        
        # Sample tracks to get the required number of track points
        tracks, visibility = self._sample_tracks_random(tracks, visibility, self.num_track_ids)
        
        return {
            'video': vids,
            'tracks': tracks, 
            'visibility': visibility,
            'task_emb': task_emb
        }

if __name__ == "__main__":
    dataset = MammalNetDataset(tracks_dir="/home/neerja/tapnet-neerja/lion_dt_animal_320_points_10_seconds_256x256/tracks", 
                               videos_dir="/datasets/mammal_net/current/full_videos/", 
                               img_size=(128, 128), 
                               num_track_ts=16, 
                               num_track_ids=32, 
                               frame_stack=1, 
                               cache_all=False, 
                               cache_image=False, 
                               num_demos=0.1, 
                               aug_prob=0.5)
    print(len(dataset))
    print(dataset[0])
    import ipdb; ipdb.set_trace()
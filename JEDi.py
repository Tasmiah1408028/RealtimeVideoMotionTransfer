pip install videojedi
pip install numpy==1.24.4 pyarrow==14.0.1 scipy==1.15.3
!pip install videojedi==0.1.2
import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_video, read_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageSequence

# 🔹 Normalize filenames by removing repeated extensions like ".mp4.mp4"
def normalize_filename(filename):
    base, ext = os.path.splitext(filename)
    while ext in ['.mp4', '.png', '.gif']:  # Keep removing repeated extensions
        base, ext = os.path.splitext(base)
    return base

# 🔹 Get common filenames, ensuring normalization
def get_common_filenames(real_folder, generated_folder):
    """Finds common video filenames in both folders, ignoring extra extensions."""
    
    real_files = {normalize_filename(f) for f in os.listdir(real_folder) if f.endswith(('.mp4', '.png', '.gif'))}
    generated_files = {normalize_filename(f) for f in os.listdir(generated_folder) if f.endswith(('.mp4', '.png', '.gif'))}

    common = real_files.intersection(generated_files)
    print(f"✅ Found {len(common)} common files.")  # ✅ Debugging output
    return common

# 🔹 Base Dataset Class for both real and generated videos
class BaseVideoDataset(Dataset):
    def __init__(self, video_folder, common_files, transform=None):
        self.video_folder = video_folder
        self.video_files = [
            os.path.join(video_folder, f) for f in os.listdir(video_folder) 
            if os.path.isfile(os.path.join(video_folder, f))  # ✅ Ignore directories
            and normalize_filename(f) in common_files  # ✅ Normalize filename before checking
            and f.endswith(('.mp4', '.png', '.gif'))
        ]
        print(f"✅ Loading {len(self.video_files)} videos from {video_folder}")  # ✅ Debugging output

        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def load_gif(self, file_path):
        """Loads a GIF and converts it into a tensor of frames (T, C, H, W)."""
        gif = Image.open(file_path)  # Open GIF
        frames = [transforms.ToTensor()(frame.convert("RGB")) for frame in ImageSequence.Iterator(gif)]  
        frames = torch.stack(frames)  # Stack into (T, C, H, W)
        return frames

    def _truncate_to_multiple_of_12(self, frames):
        """Ensures the number of frames is a multiple of 12 by truncating excess frames."""
        num_frames = frames.shape[0]
        max_frames = (num_frames // 12) * 12  # Compute nearest multiple of 12
        return frames[:max_frames]  # Truncate frames

    def _pad_to_1224_frames(self, frames):
        """Ensures every video has exactly 1224 frames by padding if necessary."""
        num_frames = frames.shape[0]
        
        if num_frames < 216:
            pad_size = 216 - num_frames
            pad_frames = torch.zeros((pad_size, *frames.shape[1:]))  # Zero-padding
            frames = torch.cat((frames, pad_frames), dim=0)  # Concatenate original and padded frames
        else:
            frames = frames[:216]
    
        return frames  # Return padded frames

    def __getitem__(self, idx):
        file_path = self.video_files[idx]

        if file_path.endswith('.gif'):
            frames = self.load_gif(file_path)  # ✅ Load GIF frames correctly
        
        elif file_path.endswith('.png'):
            img = read_image(file_path)  # Shape: (C, H, W)
            frame_width = 256  # Assuming each frame is 256px wide
            num_frames = img.shape[2] // frame_width
            frames = torch.stack([img[:, :, i * frame_width:(i + 1) * frame_width] for i in range(num_frames)])

        elif file_path.endswith('.mp4'):
            video, _, _ = read_video(file_path, pts_unit="sec")  # Shape: (T, H, W, C)
            video = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
            frames = video

        else:
            raise ValueError(f"❌ Unsupported file format: {file_path}")

        # Normalize to [0,1]
        frames = frames.float() / 255.0  

        # Ensure the number of frames is a multiple of 12
        #frames = self._truncate_to_multiple_of_12(frames)

        # Apply padding
        frames = self._pad_to_1224_frames(frames)

        #print(f"✅ Video: {file_path}, Frames After Processing: {frames.shape[0]}")

        if self.transform:
            frames = self.transform(frames)

        return frames, 0  # Return video frames + dummy label


# 🔹 Define dataset for real videos (inherits from BaseVideoDataset)
class Real_VideoDataset(BaseVideoDataset):
    pass  # No modifications needed, inherits everything


# 🔹 Define dataset for generated videos (inherits from BaseVideoDataset)
class Generated_VideoDataset(BaseVideoDataset):
    def __init__(self, video_folder, common_files, transform=None):
        super().__init__(video_folder, common_files, transform)  # ✅ Properly initialize

    def __getitem__(self, idx):
        file_path = self.video_files[idx]

        if file_path.endswith('.gif'):
            frames = self.load_gif(file_path)  # ✅ Load GIF frames correctly
        
        elif file_path.endswith('.png'):
            img = read_image(file_path)  # Shape: (C, H, W)
            frame_width = 256
            num_frames = img.shape[2] // frame_width
            frames = torch.stack([img[:, :, i * frame_width:(i + 1) * frame_width] for i in range(num_frames)])

        elif file_path.endswith('.mp4'):
            video, _, _ = read_video(file_path, pts_unit="sec")  # Shape: (T, H, W, C)
            video = video.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
            # Determine cropping based on video shape
            _, _, height, width = video.shape  # Get dimensions
    
            if height == 1280 and width == 256:
                # Crop height from 768 to 1024 (keep full width)
                video = video[:, :, 768:1024, :]
            else:
                # Crop width from 768 to 1024 (keep full height)
                video = video[:, :, :, 768:1024]
            #video = video[:, :, :, 768:1024]  # Crop width from 768 to 1024
            frames = video

        else:
            raise ValueError(f"❌ Unsupported file format: {file_path}")

        # Normalize to [0,1]
        frames = frames.float() / 255.0  

        # Ensure the number of frames is a multiple of 12
        #frames = self._truncate_to_multiple_of_12(frames)

        # Apply padding
        frames = self._pad_to_1224_frames(frames)

        #print(f"✅ Video: {file_path}, Frames After Processing: {frames.shape[0]}")

        if self.transform:
            frames = self.transform(frames)

        return frames, 0  # Return video frames + dummy label


# 🔹 Define paths to real and generated video folders
real_folder = "./Training_Prediction/FOMM/datasets/vox/test_recon"  # Change this to actual folder
generated_folder = "/home/jovyan/srinjoy-vol/Motion-Transfer-Keypoints-Prediction/Keypoints_Prediction/log/test-reconstruction-vox/prediction/VRNN_voxtransfer_12-12_pytorch"  # Change this to actual folder

# 🔹 Get common filenames after normalizing
common_files = get_common_filenames(real_folder, generated_folder)

# 🔹 Define optional transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize if needed
])

# 🔹 Create DataLoaders only for matched videos
real_videos = Real_VideoDataset(video_folder=real_folder, common_files=common_files, transform=transform)
generated_videos = Generated_VideoDataset(video_folder=generated_folder, common_files=common_files, transform=transform)

real_loader = DataLoader(real_videos, batch_size=4, shuffle=False)
generated_loader = DataLoader(generated_videos, batch_size=4, shuffle=False)

print("✅ Data loading complete! Only matching videos are kept, and GIFs are properly handled.")

from videojedi import JEDiMetric

# Step 1: Initialize JEDi
jedi = JEDiMetric(feature_path=".", model_dir="./videojedi_models")  # You can specify a different model directory if needed

# Step 2: Extract features from real and generated videos
jedi.load_features(train_loader=real_loader, test_loader=generated_loader, num_samples=44)  # Adjust num_samples if needed

# Step 3: Compute JEDi Score
jedi_score = jedi.compute_metric()
print(f"JEDi Score: {jedi_score}")
# %%
import os
import pandas as pd
import torch
import time
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from transformers import BertTokenizer
from torchvision.transforms import Compose, Lambda, Resize, Normalize, ColorJitter

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


class DeepFakeDataset(Dataset):
    def __init__(self, csv_file, video_root, text_transforms, video_transforms, num_frames=8, sampling_rate=8, frames_per_second=30):
        self.data = pd.read_csv(csv_file)
        self.video_root = video_root
        self.text_transforms = text_transforms
        self.video_transforms = video_transforms
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frames_per_second = frames_per_second

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_path = os.path.join(self.video_root, self.data.iloc[index]['video_id']+".mp4")
        label = self.data.iloc[index]['label']
        text = self.data.iloc[index]['text']

        # # Load video using PyTorchVideo
        # video = EncodedVideo.from_path(video_path)

        # # Get video duration and calculate the step size for frame sampling
        # duration = video.duration
        # step = duration / self.num_frames

        # # Sample frames at regular intervals
        # video_data = []
        # for i in range(self.num_frames):
        #     start_sec = i * step
        #     end_sec = start_sec + step
        #     clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        #     video_data.append(self.video_transforms(clip))

        # # Stack the sampled frames
        # # video_data = torch.stack(video_data)

        # # Stack the sampled frames
        # # Extract video tensors from each dictionary
        # video_tensors = [item['video'] for item in video_data]

        # # Stack video tensors along the frames dimension
        # stacked_video = torch.stack(video_tensors).squeeze(2).permute(1, 0, 2, 3)

        clip_duration = (num_frames * sampling_rate)/frames_per_second
        start_sec = 0
        end_sec = start_sec + clip_duration

        # Initialize an EncodedVideo helper class and load the video
        video = EncodedVideo.from_path(video_path)

        # video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        video_data = video.get_clip(start_sec=start_sec, end_sec=video.duration)

        # Apply a transform to normalize the video input
        video_data = transform(video_data)


        # Apply text transforms
        # text_data = self.text_transforms(text)
        text_data = text

        return {
            'video': video_data['video'],
            'text': text_data,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create instances of text and video transforms
text_transforms = Compose([
    BertTokenizer.from_pretrained('bert-base-uncased'),
    # Add more text transformations as needed
])

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# Create an instance of the dataset
csv_file = '../train.csv'
video_root = '../train'

dataset = DeepFakeDataset(csv_file, video_root, text_transforms, transform, num_frames, sampling_rate, frames_per_second)

# Create a dataloader
batch_size = 4
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

cnt = 0
video_train=list()
start_time = time.time()
for data in dataloader:
    video_train.append(data['video'])

    cnt += 1

    if cnt == 3:
        break
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Function execution time: {elapsed_time} seconds")
print(video_train[0].shape)
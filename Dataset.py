import pandas as pd
import torch
from torch.utils.data import Dataset

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from pytorchvideo.data.encoded_video import EncodedVideo

def preprocess_video(input_folder, output_folder):

    # Create output folders if they do not exist
    video_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    preprocessing_types = ['preprocessed', 'normalized', 'grayscale', 'noisy', 'blurred', 'sharpened']

    for preprocessing_type in preprocessing_types:

        folder_path = os.path.join(output_folder, preprocessing_type)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    for video_file in video_files:
        # Open the video file
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

    # Convert the frame to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def preprocess_text(text):

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize text
    tokens = bert_tokenizer.tokenize(text)

    # Add special tokens [CLS] and [SEP]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # Convert tokens to token IDs
    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    original_input_len = len(input_ids)

    # Pad or truncate input IDs to a fixed length
    max_length = 512
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + [0] * padding_length  # Padding token ID for BERT

    # Create attention mask
    attention_mask = [1]*original_input_len  + [0]*padding_length # 1 for real tokens, 0 for padding tokens

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # Add batch dimension

    return input_ids, attention_mask

class DeepFakeDataset(Dataset):
    
    def __init__(self, video_path_file, text_csv_file, text_transforms=None, video_transforms=None, num_frames=8):
        self.video_annotation = pd.read_csv(video_path_file)
        self.text_df = pd.read_csv(text_csv_file)
        self.text_transforms = text_transforms
        self.video_transforms = video_transforms
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_annotation)

    def __getitem__(self, index):

        video_path = self.video_annotation.iloc[index]['video_path']
        label = self.video_annotation.iloc[index]['label']
        text = self.text_df.iloc[index]['text']
        # print('Before Text reduction')
        # print(f'len of text: {len(text)}')
        # print(text)
        text = text[:512]
        # print('After Text reduction')
        # print(f'len of text: {len(text)}')
        # print(text)
        # print('\n')

        try:
            # Load video using PyTorchVideo
            video = EncodedVideo.from_path(video_path)

            # Get video duration and calculate the step size for frame sampling
            duration = video.duration
            step = duration / self.num_frames
            # print(f'Video length: {duration}')

            # Sample frames at regular intervals
            video_data = []
            for i in range(self.num_frames):
                start_sec = i * step
                end_sec = start_sec + step
                clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)
                # print(f'clip shape: {clip['video'].shape}')
                transformed_clip = self.video_transforms(clip['video'])
                # print(f'Transformed Clip: {transformed_clip.shape}')
                video_data.append(transformed_clip)

            # # Stack the sampled frames
            video_data = torch.stack(video_data)
            # print(video_data)
            # print(f'Video_data shape: {video_data.shape}')
            # print('--------------------------------------------\n')

            # Medium example
            # start_time = 0
            # clip_duration = int(video.duration)
            # end_sec = start_time + clip_duration
            # video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
            # video_data = self.video_transforms(video_data['video'])
 

        except Exception as e:
            print(f'Error Processing video {video_path}: {e}')
            # print(f'Clip Duration: {clip_duration}')
            # print(f'Video Duration: {video.duration}')

        # Apply text transforms
        id, mask = None, None

        # print('Entering text transform')
        # if self.text_transforms:
        #     id, mask = self.text_transforms(text)
        # else:
        #     pass
        id, mask = preprocess_text(text=text)

        # print(f'id shape = {id.shape}')
        # print(f'id = {id}')
        # print(f'mask shape = {mask.shape}')
        # print(f'mask = {mask}')
        # print('End of Get item')

        return {
            'id':id,
            'mask':mask,
            'video': video_data,
            'label': torch.tensor(label, dtype=torch.long)
        }
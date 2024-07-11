import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50

class HierarchicalEncodingNetwork(nn.Module):

    def __init__(self, mcan_model, bert, resnet, groups=3) -> None:
        super(HierarchicalEncodingNetwork, self).__init__()

        self.bert = bert
        self.resnet = resnet
        
        # Multimodal Contextual Attention Network model
        self.mcan = mcan_model
        
        # Number of groups to split BERT layers
        self.layer_groups = groups

        # 2D Conv Layer to transform 2048 img features to 768
        # self.conv2d = nn.Conv2d(in_channels=)

        self.output_dim = 2 * self.layer_groups * 768

        # self.classifier = nn.Linear(output_dim*groups, 2)

        self.classifier = nn.Sequential(
                        nn.Linear(self.output_dim, 1),
                        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids, attention_mask, video):

        print(f'input_ids shape: {input_ids.shape}')
        print(f'Attention mask shape: {attention_mask.shape}')
        print(f'In hmcan, video.shape: {video.shape}')
        # print('Enter hmcan forward')

        # print('Before Squeezing Input ID/attn mask shape')
        # print(f'input ID shape: {input_ids.shape}')
        # print(f'Mask shape: {attention_mask.shape}')

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        print('IN hmcan, after squeezing')
        print(f'input_ids shape: {input_ids.shape}')
        print(f'Attention mask shape: {attention_mask.shape}')

        # print('After Squeezing Input ID/attn mask shape')
        # print(f'input ID shape: {input_ids.shape}')
        # print(f'Mask shape: {attention_mask.shape}')

        # Extract hidden states from BERT
        hidden_states = self.bert(input_ids, attention_mask=attention_mask).hidden_states

        # print(f'hidden_states shape: {hidden_states.shape}')

        # print('Finished with BERT')
        
        # change video format from (B, T, C, 1, H, W) to (B, C, T, H, W)
        video = video.squeeze(3).permute(0, 2, 1, 3, 4)
        print(f'After hidden states, video shape: {video.shape}')

        # Resnet 3D expects (B, C, T, H, W) and produces (B, C, T, H, W)
        video_features = self.resnet(video)
        print(f'After resnet, video shape: {video_features.shape}')

        # Get the original shape of the tensor
        B, C, T, H, W = video_features.shape

        # Reshape the tensor to (B, C, T*H*W)
        video_features = video_features.view(B, C, T*H*W)
        print(f'video_features after reshape(B, C, T*H*W): {video_features.shape}')
        video_features = video_features.permute(0, 2, 1)
        print(f'After resizing, video shape: {video_features.shape}')

        # Now video_features should be [B, 512, 768]

        # input2 = img_features.view(img_features.size(0), img_features.size(1), -1)  # Changing to [batch, channels, height * width]
        # input2 = input2.permute(0, 2, 1)  # Rearrange to [batch, height * width, channels]

        # print('After resnet')
        # print(f'Img feature size = {input2.shape} \n')

        # Group BERT layers' outputs and process each group with the image features through MCAN
        grouped_text_features = self._group_bert_outputs(hidden_states)

        # print(f'Getting grouped text features size: {grouped_text_features[0].shape} \n')
        # print('Inserting text and image into mcan\n')

        combined_outputs = []
        
        for text_features in grouped_text_features:
            mcan_output = self.mcan(text_features, video_features)
            combined_outputs.append(mcan_output)

        final_features = torch.cat(combined_outputs, dim=-1)
        output = self.classifier(final_features)
        output = self.sigmoid(output)

        return output
    
    def _group_bert_outputs(self, hidden_states):
        # Simplified grouping strategy
        step = len(hidden_states) // self.layer_groups

        return [torch.mean(torch.stack(hidden_states[i*step:(i+1)*step]), dim=0) for i in range(self.layer_groups)]
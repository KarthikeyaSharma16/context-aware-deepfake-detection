import torch
from transformers import BertTokenizer

class Tokenizer:
    def __init__(self, max_length=512):
        self.max_len = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, text):
        print('Enter Tokenizer')
        # Tokenize text
        tokens = self.bert_tokenizer.tokenize(text)
        
        # Add special tokens [CLS] and [SEP]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert tokens to token IDs
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        
        # Pad or truncate input IDs to a fixed length
        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + [0] * padding_length  # Padding token ID for BERT
        
        # Create attention mask
        attention_mask = [1] * len(tokens) + [0] * padding_length  # 1 for real tokens, 0 for padding tokens
        
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        print('End of Tokenizer')
        return input_ids, attention_mask
    

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
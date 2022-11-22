import os
import torch
import pandas as pd
from PIL import Image
from PIL import ImageFile
from torch.utils.data.dataset import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-defined hypaparameters
Image.MAX_IMAGE_PIXELS = 1000000000
dataset = 'memotion'

memotion_sentiment_2_index = {
    "very_negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very_positive": 4
}

class MemotionDataset(Dataset):
    """Memotion dataset proposed by SEMEVAL-2020"""
    
    def __init__(self, root_dir, split, model_name, max_len, tokenizer, transform=None):
        # Metadata
        self.full_data_path = os.path.join(root_dir, dataset) + "/{}".format(split) + "/{}.csv".format(split)
        self.data_dict = pd.read_csv(self.full_data_path, sep=',')
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        self.split = split
        # BERT tokenizer
        self.tokenizer = tokenizer
        self.max_len = max_len
           
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir+'/'+self.dataset + "/{}".format(self.split) + '/images/' + self.data_dict.iloc[idx, 1]
        image = Image.open(img_name).convert('RGB')
        label = memotion_sentiment_2_index[self.data_dict.iloc[idx, 8]]

        text = str(self.data_dict.iloc[idx, 2])
        text_encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'input_ids': text_encoded['input_ids'].flatten(),
                  'attention_mask': text_encoded['attention_mask'].flatten(),
                  "label": torch.tensor(label, dtype=torch.long)}
        return sample

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from utils.dataset.multi_category_memotion import MultiCategoryeMemotionEvalDataset

index_2_sentiment = ["happiness", "love", "anger", "sorrow", "fear", "hate", "surprise"]

activation = {}

@torch.no_grad()
def get_activation(name1, name2):
    def hook(model, input, output):
        activation[name1] = output.last_hidden_state.detach()
        activation[name2] = output.pooler_output.detach()
    return hook

@torch.no_grad()
def get_memes_to_prob_dist_map(args, device, meme_text_list: list):
    """
    Use pretrained multi category meme model to process meme images with their texts
    :param args
    :param meme_text_list: [{"meme_filename": "image_ (0).jpg", "text": "\"That moment after you throw up and your friend asks you \"\"YOU GOOD BRO?\"\" I'M FUCKIN LIT"}]
    :return meme_filename_to_prob_dist: dict (map from meme filename to its probability distribution over 7 sentiment classes)
    """
    # Load tokenizer and pretrained model
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    model = torch.load(os.path.join(args.model_dir, args.pretrained_multi_category_meme_model),
                       map_location=torch.device(device))
    bert = torch.load(os.path.join(args.model_dir, args.pretrained_bert), map_location=torch.device(device))
    # print(model)
    feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', args.cnn_model, weights='VGG16_Weights.IMAGENET1K_V1')

    # Init dataset
    dataset = MultiCategoryeMemotionEvalDataset(
        args.meme_image_dir,
        meme_text_list,
        args.max_token_length,
        tokenizer,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    )
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, generator=torch.Generator(device=device))
    meme_filename_to_prob_dist = {}
    for i_batch, data_batch in enumerate(dataset_loader):
        meme_filenames = data_batch["meme_filename"]
        input_ids = data_batch["input_ids"]
        images = data_batch['image']
        attention_mask = data_batch['attention_mask']

        if device == 'cude':
            with torch.cuda.device(0):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                targets = targets.cuda()
                images = images.cuda()

        if images.size()[0] != input_ids.size()[0]:
            continue

        feature_images = feature_extractor.features(images)
        feature_images = feature_extractor.avgpool(feature_images)
        feature_images = torch.flatten(feature_images, 1)
        feature_images = feature_extractor.classifier[0](feature_images)

        bert.bert.register_forward_hook(get_activation('last', 'pool'))
        outputs = bert(input_ids, attention_mask)

        outputs = model(
            last_hidden=activation['last'],
            pooled_output=activation['pool'],
            feature_images=feature_images
        ).tolist()

        # Collect the results into dictionary
        print(meme_filenames)
        print(outputs)
        # _, preds = torch.max(outputs, dim=1)
        # for pred in preds:
        #     print(index_2_sentiment[pred])
        for idx, filename in enumerate(meme_filenames):
            meme_filename_to_prob_dist[filename] = outputs[idx]

        with open("data/meme_filename_to_prob_dist.json", "w") as outfile:
            json.dump(meme_filename_to_prob_dist, outfile)


def main(args, device):
    # TODO preprocess meme images
    meme_text_list = [
        {"meme_filename": "image_ (1093).jpg", "text": "OMG?!They told me this was a solo protrait!"},
        {"meme_filename": "image_ (109).jpg", "text": "NEXT STOP: KNOWLEDGE"}
    ]
    get_memes_to_prob_dist_map(args, device, meme_text_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme_image_dir", type=str, default="data/Multi_category_Meme/images/",
                        help="where meme images are stored")

    parser.add_argument("--model_dir", type=str, default="pretrained_models",
                        help="where pretrained model is stored")
    parser.add_argument("--pretrained_multi_category_meme_model", type=str, default="multi_category_meme_model.pt",
                        help="filename of pretrained multi category meme model")
    parser.add_argument("--pretrained_bert", type=str, default="pretrained_albert.pt",
                        help="filename of pretrained bert model")

    parser.add_argument("--max_token_length", type=int, default=50,
                        help="max token length of meme text")
    parser.add_argument('--cnn_model', type=str, default="vgg16",
                        help='pretrained CNN to use for image feature extraction')
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args, device)

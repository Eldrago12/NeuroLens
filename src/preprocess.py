import os
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

DATA_DIR = '/data/flickr8k'
IMAGE_DIR = os.path.join(DATA_DIR, 'Images')
CAPTION_FILE = os.path.join(DATA_DIR, 'captions.txt')
MAX_VOCAB_SIZE = 5000

def load_captions(filename):
    captions_dict = {}
    with open(filename, 'r') as file:
        header = next(file)

        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            image_file, caption = parts[0], parts[1]
            caption = caption.lower().translate(str.maketrans('', '', string.punctuation))
            caption = 'startseq ' + caption + ' endseq'

            if image_file not in captions_dict:
                captions_dict[image_file] = []
            captions_dict[image_file].append(caption)
    return captions_dict

def build_tokenizer(captions_dict, max_vocab_size=MAX_VOCAB_SIZE):
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='unk')
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

def max_caption_length(captions_dict):
    all_captions = []
    for captions in captions_dict.values():
        all_captions.extend(captions)
    return max(len(caption.split()) for caption in all_captions)

def extract_image_features(image_dir, image_files):
    model = InceptionV3(weights='imagenet')
    model = tf.keras.Model(model.input, model.layers[-2].output)
    features = {}
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        img = load_img(img_path, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        features[img_name] = feature.squeeze()
    return features


if __name__ == '__main__':
    captions_dict = load_captions(CAPTION_FILE)
    print("Total images with captions:", len(captions_dict))
    tokenizer = build_tokenizer(captions_dict)
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    max_len = max_caption_length(captions_dict)
    print("Max caption length:", max_len)
    image_files = list(captions_dict.keys())
    features = extract_image_features(IMAGE_DIR, image_files)
    with open('image_features.pkl', 'wb') as f:
        pickle.dump(features, f)

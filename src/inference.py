import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import io
from flask import Flask, request, jsonify
from PIL import Image
import sys
import keras

sys.modules['keras.src'] = keras
sys.modules['keras.src.legacy'] = keras

app = Flask(__name__)

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config

class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            unroll=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=hidden)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'embedding_dim': self.embedding.input_dim,
            'units': self.units,
            'vocab_size': self.fc2.units,
        })
        return config

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, decoder, **kwargs):
        super(ImageCaptioningModel, self).__init__(**kwargs)
        self.decoder = decoder
    def call(self, inputs):
        features, seq = inputs
        batch_size = tf.shape(features)[0]
        hidden = self.decoder.reset_state(batch_size)
        features = tf.expand_dims(features, 1)
        dec_input = tf.fill([batch_size, 1], tokenizer.word_index['startseq'])
        outputs = []
        seq_length = tf.constant(max_length, dtype=tf.int32)
        for i in tf.range(seq_length):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            outputs.append(predictions)
            dec_input = tf.gather(seq, [i], axis=1)
        return tf.stack(outputs, axis=1)
    def get_config(self):
        config = super(ImageCaptioningModel, self).get_config()
        config.update({'decoder': self.decoder.get_config()})
        return config
    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop('decoder', None)
        decoder = Decoder.from_config(decoder_config) if decoder_config else None
        return cls(decoder, **config)

with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
max_length = 34
vocab_size = len(tokenizer.word_index) + 1
EMBEDDING_DIM = 256
UNITS = 512

decoder = Decoder(EMBEDDING_DIM, UNITS, vocab_size)
final_model = ImageCaptioningModel(decoder)
dummy_features = tf.random.uniform((1, 2048))
dummy_seq = tf.random.uniform((1, max_length), maxval=vocab_size, dtype=tf.int32)
_ = final_model((dummy_features, dummy_seq))
final_model.load_weights('models/image_captioning_model.h5')
decoder = final_model.decoder

def preprocess_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((299, 299))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features(image_bytes):
    """Extract image features using InceptionV3 (penultimate layer)."""
    inception = InceptionV3(weights='imagenet')
    model_inception = tf.keras.Model(inception.input, inception.layers[-2].output)
    x = preprocess_image_from_bytes(image_bytes)
    feature = model_inception.predict(x)
    return feature.squeeze()

def generate_caption(image_bytes):
    features = extract_features(image_bytes)
    features = tf.expand_dims(features, 0)
    features = tf.expand_dims(features, 1)

    hidden = decoder.reset_state(1)
    dec_input = tf.fill([1, 1], tokenizer.word_index['startseq'])

    result = []
    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()

        predicted_word = None
        for word, idx in tokenizer.word_index.items():
            if idx == predicted_id:
                predicted_word = word
                break
        if predicted_word is None:
            break
        result.append(predicted_word)
        if predicted_word == 'endseq':
            break
        dec_input = tf.fill([1, 1], predicted_id)

    if result and result[-1] == 'endseq':
        caption = ' '.join(result[1:-1])
    else:
        caption = ' '.join(result[1:])
    return caption

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    image_bytes = file.read()
    caption = generate_caption(image_bytes)
    return jsonify({'caption': caption})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

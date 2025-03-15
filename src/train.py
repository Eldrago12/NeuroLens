import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import load_captions, max_caption_length

TOKENIZER_PATH = 'tokenizer.pkl'
IMAGE_FEATURES_PATH = 'image_features.pkl'
CAPTION_FILE = '/kaggle/input/flickr8k/captions.txt'

BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 256
UNITS = 512
EPOCHS = 10

strategy = tf.distribute.MirroredStrategy()
print("Number of devices (GPUs):", strategy.num_replicas_in_sync)

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

with open(IMAGE_FEATURES_PATH, 'rb') as f:
    image_features = pickle.load(f)

captions_dict = load_captions(CAPTION_FILE)
max_length = max_caption_length(captions_dict)

def create_training_data(captions_dict, image_features):
    img_names = []
    captions = []
    for img_name, caps in captions_dict.items():
        if img_name not in image_features:
            continue
        for cap in caps:
            img_names.append(img_name)
            captions.append(cap)
    return img_names, captions

img_names, all_captions = create_training_data(captions_dict, image_features)

sequences = tokenizer.texts_to_sequences(all_captions)
cap_vector = pad_sequences(sequences, padding='post', maxlen=max_length)

features = np.array([image_features[img_name] for img_name in img_names])

dataset = tf.data.Dataset.from_tensor_slices((features, cap_vector))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dist_dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    decoder = Decoder(EMBEDDING_DIM, UNITS, vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # Force building the decoder by calling it once with dummy inputs.
    dummy_input = tf.fill([BATCH_SIZE, 1], tokenizer.word_index['startseq'])
    dummy_features = tf.random.uniform((BATCH_SIZE, 1, 2048))
    dummy_hidden = decoder.reset_state(BATCH_SIZE)
    _ = decoder(dummy_input, dummy_features, dummy_hidden)

    # Force optimizer variable creation by running a dummy update in the proper replica context.
    def dummy_update_fn():
        dummy_grads = [tf.zeros_like(var) for var in decoder.trainable_variables]
        optimizer.apply_gradients(zip(dummy_grads, decoder.trainable_variables))
    strategy.run(dummy_update_fn)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor, target):
    with tf.GradientTape() as tape:
        batch_size = tf.shape(img_tensor)[0]
        hidden = decoder.reset_state(batch_size)
        features_batch = tf.expand_dims(img_tensor, 1)
        dec_input = tf.fill([batch_size, 1], tokenizer.word_index['startseq'])

        loss = 0.0
        seq_length = tf.constant(max_length, dtype=tf.int32)
        for i in tf.range(1, seq_length):
            predictions, hidden, _ = decoder(dec_input, features_batch, hidden)
            current_target = tf.gather(target, [i], axis=1)
            loss += loss_function(current_target, predictions)
            dec_input = current_target

        total_loss = loss / tf.cast(seq_length, tf.float32)

    trainable_variables = decoder.trainable_variables
    gradients = tape.gradient(total_loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return total_loss

@tf.function
def distributed_train_step(dist_inputs):
    img_tensor, target = dist_inputs
    per_replica_losses = strategy.run(train_step, args=(img_tensor, target))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

print("Running a dummy training step...")
dummy_batch = next(iter(dist_dataset))
dummy_loss = distributed_train_step(dummy_batch)
print("Dummy step loss:", dummy_loss.numpy())

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, decoder=decoder)

for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    for batch_data in dist_dataset:
        loss_value = distributed_train_step(batch_data)
        total_loss += loss_value
        num_batches += 1
    epoch_loss = total_loss / num_batches
    checkpoint.save(file_prefix=checkpoint_prefix)
    print(f"Epoch {epoch+1} Loss {epoch_loss:.4f}")

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, decoder):
        super(ImageCaptioningModel, self).__init__()
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

final_model = ImageCaptioningModel(decoder)
sample_features = tf.random.uniform((BATCH_SIZE, 2048))
sample_seq = tf.random.uniform((BATCH_SIZE, max_length), maxval=vocab_size, dtype=tf.int32)
_ = final_model((sample_features, sample_seq))

final_model.save('/kaggle/working/image_captioning_model.h5')
print("Model saved as /kaggle/working/image_captioning_model.h5")

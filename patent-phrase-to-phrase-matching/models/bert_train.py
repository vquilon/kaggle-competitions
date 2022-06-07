import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

from patent_phrase_similarity.data.transformation.cpc_datasets import Datasets, CPCDatasets

max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

# Labels in our dataset.
labels = ["0.0", "0.25", "0.5", "0.75", "1.0"]


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=batch_size,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            truncation=True,
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


def create_bert_model():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # Encoded token ids from BERT tokenizer.
        input_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks indicates to the model which tokens should be attended to.
        attention_masks = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids are binary masks identifying different sequences in the model.
        token_type_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        # Loading pretrained BERT model.
        bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False

        bert_output = bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        sequence_output = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
        concat = tf.keras.layers.concatenate([avg_pool, max_pool])
        dropout = tf.keras.layers.Dropout(0.3)(concat)
        output = tf.keras.layers.Dense(len(labels), activation="softmax")(dropout)
        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )


    print(f"Strategy: {strategy}")
    model.summary()

    return model, bert_model


def check_similarity(_model, anchor, target):
    sentence_pairs = np.array([[str(anchor), str(target)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    test_data = test_data.with_options(options)

    proba = _model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba


if __name__ == '__main__':
    datasets = Datasets()
    cpc_datasets = CPCDatasets()
    train_orig_df = datasets.get_train_df()
    test_df = datasets.get_test_df()
    cpc_train_df = cpc_datasets.merge_with_df(train_orig_df)

    # Se divide el de train en train y valid
    # train_df, valid_df = train_df.randomSplit([0.8, 0.2], seed=42)
    msk_80 = np.random.rand(len(cpc_train_df)) < 0.8
    _train_df = cpc_train_df[msk_80]
    _valid_df = cpc_train_df[~msk_80]

    y_train = tf.keras.utils.to_categorical(_train_df.score, num_classes=len(labels))
    y_val = tf.keras.utils.to_categorical(_valid_df.score, num_classes=len(labels))
    # y_test = tf.keras.utils.to_categorical(test_df.score, num_classes=len(labels))

    train_data = BertSemanticDataGenerator(
        _train_df[["anchor", "target"]].values.astype("str"),
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_data = BertSemanticDataGenerator(
        _valid_df[["anchor", "target"]].values.astype("str"),
        y_val,
        batch_size=batch_size,
        shuffle=False,
    )

    model, bert_model = create_bert_model()

    # TRAIN
    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
    )
    # Unfreeze the bert_model.
    bert_model.trainable = True
    # Recompile the model to make the change effective.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
    )

    # TEST
    # test_data = BertSemanticDataGenerator(
    #     test_df[["anchor", "target"]].values.astype("str"),
    #     y_test,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )
    # model.evaluate(test_data, verbose=1)
    # PREDICT on test

    test_df['score'] = model.predict(test_df[["anchor", "target"]].values.astype("str"))
    print(test_df)

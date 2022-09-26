from transformers import AutoTokenizer, TFGPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, create_optimizer
import tensorflow as tf
from transformers.keras_callbacks import PushToHubCallback
from dataset import tokenizer, context_length, tokenized_datasets
import os

DEVICE = "tpu"
BATCH_SIZE = 1024
EPOCHS = 10

def tpu_init():
    assert 'COLAB_TPU_ADDR' in os.environ, 'Missing TPU; did you request a TPU in Notebook Settings?'

    if 'COLAB_TPU_ADDR' in os.environ:
        TF_MASTER = 'grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])
    else:
        TF_MASTER=''

    tpu_address = TF_MASTER

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    print("Number of devices: ", len(tf.config.list_logical_devices('TPU')))

    strategy = tf.distribute.TPUStrategy(resolver)

    return strategy




config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="tf")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=BATCH_SIZE,
)
tf_eval_dataset = tokenized_datasets["valid"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=BATCH_SIZE,
)

num_train_steps = len(tf_train_dataset)
strategy = tpu_init()
with strategy.scope():
    optimizer, schedule = create_optimizer(
        init_lr=5e-5,
        num_warmup_steps=1_000,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model = TFGPT2LMHeadModel(config)
    model(model.dummy_inputs)  # Builds the model
    model.compile(optimizer=optimizer)
model.summary()

# Train in mixed-precision float16
# tf.keras.mixed_precision.set_global_policy("mixed_float16")

callback = PushToHubCallback(output_dir="codeparrot-ds-gpt2", tokenizer=tokenizer)

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=EPOCHS, callbacks=[callback])
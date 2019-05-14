# BERT with History Answer Embedding for Conversational Question Answering

## History Answer Embedding

Incorporate history turns with history answer embedding (HAE) to a [BERT](https://github.com/google-research/bert) based machine comprehension model. Paper link: to be updated

### Run

1. Download the `BERT-base Uncased` model [here](https://github.com/google-research/bert).
2. Download the [QuAC](http://quac.ai/) data.
3. Configurate the directories for the BERT model and data in `cqa_flags.py`. Also, specify a cache directory in it.
4. Run (with optimal hyper-parameters)

```
python hae.py \
    --output_dir=OUTPUT_DIR  \
    --history=6 \
    --num_train_epochs=3.0 \
    --train_steps=24000 \
    --max_considered_history_turns=11 \
    --learning_rate=3e-05 \
    --warmup_proportion=0.1 \
    --evaluation_steps=1000 \
    --evaluate_after=18000
```

### Some program arguments

Program arguments can be set in `cqa_flgas.py`. Alternatively, they could be specified at running by command line arguments. Most of the arguments are self-explanatory. Here are some selected arguments:

* `num_train_epochs` , `train_steps`,`learning_rate` ,`warmup_proportion`: the learning rate follow a schedule of warming up to the specified larning rate and then decaying. This schedule is described in the transformer paper. Our model trains for `train_steps` instead of full `num_train_epochs` epochs. 
* `load_small_portion` . Set to `True` for loading a small portion of the data for testing purposed when we are developing the model. Set to `False` to load all the data when running the model.
* `cache_dir`. When we run the model for the first time, it preprocesses the data and saves it in a cache directory. After that, the model reads the propocessed data from the cache.

* `max_considered_history_turns` and `history`. We only consider `max_considered_history_turns` previous turns when preprocessing the data. This is typically set to 11, meaning that all previous turns are under consideration (for QuAC). We incorporate `history` previous turns via history answer embedding. This can be tunned. Current results suggest 5 or 6 give the best performance.


### Scripts

* `hae.py`. Entry code.
* `cqa_supports.py`. Utility functions.
* `scorer.py`. Official evaluation script for QuAC.

Most other files are for BERT.


### Environment

Tested with Python 3.6.7 and TensorFlow 1.8.0

### Current results:
HAE with 6 histories gives f1 of 63.1 on validation data.

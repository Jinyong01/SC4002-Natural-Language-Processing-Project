# SC4002-Natural-Language-Processing-Project

SC4002 Natural Language Processing (AY26-27 Sem 1) group assignment, Group 10.

The system reads an English assistant request and does two things with one joint model:

- **Intent classification**: picks one of the 60 MASSIVE intents (e.g. `alarm_set`).
- **Slot filling**: extracts `[slot type : span]` pairs from 55 slot types (e.g. `[time : five am]`).

It is built in two stages. Stage 1 is a baseline written without AI help. Stage 2 improves on it in collaboration with an AI assistant. No pretrained models are used apart from word vectors, and every system stays under the 20M parameter cap (word vectors included).

## Results

Scored with the same `evaluate()` function in `main.ipynb`. For each model, the saved checkpoint is the epoch with the best dev **combined** score, (intent accuracy + slot micro-F1) / 2. The test split is used only for the final scores.

| Model | Split | Total params | Trainable params | Intent accuracy (%) | Slot micro-F1 (%) | Epochs (selected / trained) |
|---|---|---|---|---|---|---|
| Stage 1: 2-layer BiLSTM | Dev | 4,398,667 | 4,398,667 | 85.15 | 78.21 | 22 / 30 |
| Stage 1: 2-layer BiLSTM | Test | 4,398,667 | 4,398,667 | 84.16 | 73.16 | 22 / 30 |
| Stage 2: GloVe + char-CNN BiLSTM-CRF | Dev | 16,116,755 | 3,834,155 | 87.56 | 81.44 | 26 / 34 |
| Stage 2: GloVe + char-CNN BiLSTM-CRF | Test | 16,116,755 | 3,834,155 | 86.15 | 79.24 | 26 / 34 |

Across three seeds, the Stage-2 model scores 86.42 ± 0.24 test intent accuracy and 79.35 ± 0.14 test slot micro-F1.

## Models

**Stage 1.** A joint BiLSTM: 300-dim word embedding (initialised from GloVe and fine-tuned), 2-layer BiLSTM with 256 hidden units per direction, a max-pooled intent head and a per-token BIO slot head. Trained with intent cross-entropy plus slot cross-entropy.

- Adam, learning rate 1e-3, batch size 32, dropout 0.3, gradient clipping at 5.0.
- Up to 30 epochs. Training stops after 10 epochs with no better dev combined score.

**Stage 2.** Changes made after diagnosing the baseline's errors on dev:

- Frozen GloVe 6B 300d vectors for the train vocabulary plus the 40k most frequent GloVe words, so many unseen dev/test words still get a meaningful vector.
- A trainable 100-dim word embedding next to GloVe.
- A character CNN per word (50-dim characters, 100 filters, width 3), for names and rare words.
- Word dropout (10%) on the trainable embedding, so the model learns what `<unk>` looks like.
- Attention pooling for the intent, with the utterance vector fed into every slot decision.
- A CRF slot decoder that blocks invalid BIO transitions.
- AdamW, learning rate 1e-3, weight decay 1e-4. The learning rate is halved after 4 epochs in a row with no better dev combined score.
- Up to 40 epochs. Training stops after 8 epochs with no better dev combined score.

The BiLSTM size, batch size, dropout and gradient clipping are the same as in Stage 1.

The notebook also contains ablations (one component removed at a time) and a seed-variation study. Only the CRF and the character CNN made a measurable difference on their own.

## Repository layout

| Path | Contents |
|---|---|
| `main.ipynb` | All code and outputs: data loading, preprocessing, shared evaluation, Stage 1, Stage 2, ablations and error analysis |
| `ai_collaboration.txt` | Chat history with the AI assistant used in Stage 2 (Claude, by Anthropic) |
| `checkpoints/` | Saved model checkpoints (see below) |
| `Group Assignment Instruction.pdf` | The assignment brief |

Checkpoints in `checkpoints/`:

| File | Written by (`main.ipynb` section) |
|---|---|
| `baseline_bilstm_2layers.pt` | Stage-1 model (4.2) |
| `stage2_glove_char_crf.pt` | Final Stage-2 model, seed 42 (6.4) |
| `ablation_*.pt` | The seven ablation runs (6.9) |
| `stage2_seed1.pt`, `stage2_seed2.pt` | Stage-2 model retrained with seeds 1 and 2 (6.10) |
| `baseline_bilstm.pt` | An earlier Stage-1 version. The current notebook does not create or use it |

## Running the notebook

1. Install Python 3 with `torch`, `numpy`, `pandas`, `matplotlib` and Jupyter. The notebook was last run with Python 3.14.7.
2. Download the MASSIVE 1.0 archive [`amazon-massive-dataset-1.0.tar.gz`](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz) and put it next to `main.ipynb`. The notebook extracts `1.0/data/en-US.jsonl` on its first run.
3. Download `glove.6B.zip` from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) and put `glove.6B.300d.txt` next to `main.ipynb`.
4. Run all cells from top to bottom.

The dataset archive and GloVe files are git-ignored. The official splits are used as released: 11,514 train, 2,033 dev and 2,974 test examples.

**Run time.** A CUDA GPU is recommended. The notebook uses one automatically if PyTorch can see it. On the GPU used for the saved outputs, a full run takes a little over an hour, about 67 minutes of it training:

| Part | Training time |
|---|---|
| Stage 1 (4.2) | about 2 min |
| Stage 2 (6.4) | about 6 min |
| Ablations (6.9) | about 44 min |
| Extra seeds (6.10) | about 14 min |

**Running all cells retrains every model and overwrites the files in `checkpoints/`.**

### Evaluating the saved checkpoints without retraining

Run every cell in order, but skip these:

- the training cell at the end of 4.2 (it starts with `# word embeddings start from glove (4.1)`)
- 5.1 (Stage-1 learning curves)
- 6.4 (Stage-2 training)
- 6.5 (Stage-2 learning curves)
- 6.9 and 6.10 (ablations and extra seeds)

5.2 and 6.6 then load `baseline_bilstm_2layers.pt` and `stage2_glove_char_crf.pt` and score them on dev and test. 6.7 and 6.8 build the comparison table and the error analysis. The learning curves, ablation table and seed table need the training cells, because they use the training history kept in memory.

## Contributors

| Name | GitHub |
|---|---|
| Koh Wee Xuan | [@Weexuan](https://github.com/Weexuan) |
| Seow Jia Wei Jerrald | [@jerrald-s](https://github.com/jerrald-s) |
| Woo Weng Tai | [@wengtttw](https://github.com/wengtttw) |
| Tan Jin Yong | [@Jinyong01](https://github.com/Jinyong01) |
| Tan Zhe Kai | [@tanzhekai](https://github.com/tanzhekai) |

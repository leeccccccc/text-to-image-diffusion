# text-to-image-diffusion

IE 7615 Deep Learning for AI — Generative Project.

## Milestone 1 — Baseline image → embedding pipeline

[`notebooks/01_baseline_pipeline.ipynb`](notebooks/01_baseline_pipeline.ipynb) builds the data and feature pipeline (no model training yet):

| Step | What it does |
|---|---|
| Dataset | Flickr30k subset ([`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k), parquet export), split **by image** using each record's `split` field |
| Cleaning | whitespace normalization, drop empty captions, skip unreadable images, convert non-RGB → RGB; original case & punctuation kept |
| Image → embedding | `CLIPProcessor` (224×224, CLIP normalization) + frozen `openai/clip-vit-base-patch32` image encoder → 512-d L2-normalized vector, one per image |
| Caption → tokens | GPT-2 tokenizer, `pad_token = eos_token`, truncate to 39 tokens + 1 EOS, pad to 40; `attention_mask` = 1 for text + terminating EOS |
| Test runs | 5 test images: image, dataset caption, embedding shape, GPT-2 token IDs |

### Subset

| split | images | image–caption pairs |
|---|---:|---:|
| train | 720 | 3,600 |
| val | 90 | 450 |
| test | 90 | 450 |
| **total** | **900** | **4,500** |

The fixed subset is stored in [`data/split_manifest.csv`](data/split_manifest.csv) (image_id, filename, split) and [`data/captions_clean.csv`](data/captions_clean.csv) (image_id, split, caption_idx, caption). Re-runs reuse these IDs and do not re-sample.

### Results

- [`results/milestone1/baseline_pipeline_test_runs.png`](results/milestone1/baseline_pipeline_test_runs.png) — Baseline Pipeline Test Runs (5 images)
- [`results/milestone1/samples.json`](results/milestone1/samples.json) — full details for the 5 runs (padded token IDs, attention mask, first embedding values)
- [`results/milestone1/run_summary.json`](results/milestone1/run_summary.json) — counts, models, package versions

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
```

## Run

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_baseline_pipeline.ipynb
```

or open the notebook and use *Restart & Run All*. A GPU is used if available but is not required.

- **Official mode** (default, `SMOKE_TEST = False`): enforces 1,000–5,000 pairs and writes `data/`, `features/`, `results/milestone1/`. On a fresh clone the manifest is reused and the ~116 MB of images are re-downloaded automatically.
- **Smoke test** (`SMOKE_TEST = True`, or env `M1_SMOKE_TEST=1`): 5 test images, writes only to `outputs_smoke/`.
- `RUN_EXTERNAL_DEMO = True` (off by default) additionally shows captions from an external pretrained model (`nlpconnect/vit-gpt2-image-captioning`). These are **not** produced by this pipeline.

Generated, git-ignored outputs: `data/images/`, `features/clip_{split}.pt` (`image_ids`, `embeddings [N,512]`), `features/gpt2_tokens_{split}.pt` (`image_ids`, `caption_idx`, `input_ids [M,40]`, `attention_mask [M,40]`). Captions link to embeddings via `image_id`.

## Repository layout

```
notebooks/01_baseline_pipeline.ipynb   Milestone 1 notebook (with outputs)
data/split_manifest.csv                fixed subset (image IDs + split)
data/captions_clean.csv                cleaned captions
results/milestone1/                    test-run figure and JSON summaries
requirements.txt                       tested package versions
```

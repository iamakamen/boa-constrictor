# BOA Evaluation Task – GRU Backbone Integration

This README shows the steps I followed to run the BOA compression pipeline with a GRU backbone on Google Colab. It is simple and reproducible.

***

## 1. Setup Environment

*   Use Google Colab (it provides GPU and is easy to use).
*   Python: 3.12.12
*   PyTorch: 2.10.0+cu128
*   CUDA: 12.8
*   GPU: Tesla T4

### Steps:

```bash
# Clone the repo
!git clone https://github.com/<your-username>/boa-constrictor.git boa
%cd boa
!git checkout -b feat/gru-backbone

# Install system build tools
!apt-get update -qq
!apt-get install -y -qq build-essential cmake ninja-build pkg-config

# Install PyTorch (compatible with Colab)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements (skip mamba-ssm and causal-conv1d)
!grep -v -E "^(mamba-ssm|causal-conv1d)" requirements.txt > reqs_partial.txt
!pip install -r reqs_partial.txt

# Install pybind11 (needed for codec)
!pip install pybind11
```

***

## 2. Add GRU Backbone

### Create a new file: `models/bytewise_gru.py`

*   Use `nn.Embedding`, `nn.GRU`, and `nn.Linear`
*   Add `init_stream()` and `step()` methods
*   Make sure `self.embedding` is defined

### Modify `model.py`

```python
from models.bytewise_gru import BytewiseGRU

def BoaConstrictor(d_model=256, num_layers=4, vocab_size=256, device="cuda", backbone="mamba"):
    if backbone == "gru":
        model = BytewiseGRU(embed_dim=d_model, hidden_dim=d_model, num_layers=num_layers)
        return model
```

### Update config: `experiments/cms_experiment/cms_experiment.yaml`

```yaml
model:
  backbone: gru
```

### Modify `main.py` to read backbone from config

```python
    model_backbone = None
    if isinstance(config.get('model', None), dict):
        model_backbone = config['model'].get('backbone', None)

    if model_backbone is None:
        model_backbone = "mamba"
    
    model = BoaConstrictor(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size, device=device, backbone=model_backbone)
```

***

## 3. Run Compression and Decompression

```bash
# Run compression
!python3 main.py --config experiments/cms_experiment/cms_experiment.yaml --compress-only --show-timings

# Run decompression
!python3 main.py --config experiments/cms_experiment/cms_experiment.yaml --decompress-only --show-timings
```

*   Use the CMS dataset from `experiments/cms_experiment`
*   This dataset is already included and ready to use
*   Run multiple times to check stability

***

## 4. Results

| Metric              | Value        |
| ------------------- | ------------ |
| Original size       | 49,920,000 B |
| BOA size (GRU)      | 50,155,324 B |
| LZMA size           | 15,489,644 B |
| ZLIB size           | 19,463,909 B |
| BOA ratio (GRU)     | 1.00         |
| LZMA ratio          | 3.22         |
| ZLIB ratio          | 2.56         |
| Compression speed   | 2.50 MB/s    |
| Decompression speed | 2.16 MB/s    |

***

## 5. Notes

*   GRU was not trained, so compression ratio ≈ 1.00
*   LZMA and ZLIB compressed much better
*   GRU integration worked fine, no errors
*   Throughput is okay for a single T4 GPU
*   Codec expects model to have: `embedding`, `init_stream`, and `step`


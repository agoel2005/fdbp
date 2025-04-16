---
license: apache-2.0
datasets:
- TIGER-Lab/MMEB-train
language:
- en
base_model:
- Qwen/Qwen2-VL-7B-Instruct
library_name: transformers
---

A new checkpoint trained using [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) with an enhanced training setup (LoRA tuning, batch size of 2048, maximum sub-dataset size of 100k). This model has shown significantly improved performance on MMEB & Flickr30K compared to the previous models using Phi-3.5 and llava-v1.6-mistral as backbone.

This repo contains the code and data for [VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks](https://arxiv.org/abs/2410.05160). In this paper, we focus on building a unified multimodal embedding model suitable for a wide range of tasks. Our approach is based on transforming an existing, well-trained Vision-Language Model (VLM) into an embedding model.

## Github
 - [Github](https://github.com/TIGER-AI-Lab/VLM2Vec)


## Data

Our model is being trained on MMEB-train and evaluated on MMEB-eval with contrastive learning. We only use in-batch negatives for training. 

 - Train data: https://huggingface.co/datasets/TIGER-Lab/MMEB-train
 - Eval data: https://huggingface.co/datasets/TIGER-Lab/MMEB-eval


## Performance
This model outperforms the baselines and previous version of VLM2Vec by a large margin.

| Model                                 | Classification | VQA  | Retrieval | Grounding | IND  | OOD  | Overall |
|---------------------------------------|---------------|------|-----------|-----------|------|------|---------|
| Phi-3.5-V, Full-model fine-tuned (#crop=4) | 52.8  | 50.3 | 57.8  | 72.3  | 62.8 | 47.4 | 55.9  |
| Phi-3.5-V, LoRA            | 54.8  | 54.9 | 62.3  | 79.5  | 66.5 | 52.0 | 60.1  |
| LLaVA-1.6, LoRA            | 54.7  | 50.3 | 56.2  | 64.0  | 61.0 | 47.5 | 55.0  |
| LLaVA-1.6, LoRA            | 61.2  | 49.9 | 67.4  | 86.1  | 67.5 | 57.1 | 62.9  |
| **Qwen2-VL-2B, LoRA (this model)**          | 59.0  | 49.4 | 65.4  | 73.4  | 66.0 | 52.6 | 60.1  |
| Qwen2-VL-7B, LoRA          | **62.6**  | **57.8** | **69.9**  | 81.7  | **72.2** | **57.8** | **65.8**  |

## How to use VLM2Vec
(More details please refer to our Github repo, here is just a simple demo.)

First you can clone our github
```bash
git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git
pip -r requirements.txt
```

```python
from src.model import MMEBModel
from src.arguments import ModelArguments
from src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens
from PIL import Image
import torch

model_args = ModelArguments(
    model_name='Qwen/Qwen2-VL-2B-Instruct',
    checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-2B',
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)

processor = load_processor(model_args)
model = MMEBModel.load(model_args)
model = model.to('cuda', dtype=torch.bfloat16)
model.eval()

# Image + Text -> Text
inputs = processor(text=f'{vlm_image_tokens[QWEN2_VL]} Represent the given image with the following question: What is in the image',
                   images=Image.open('figures/example.jpg'),
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.2500]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.1865]], device='cuda:0', dtype=torch.bfloat16)

```


## Citation
```
@article{jiang2024vlm2vec,
  title={VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  author={Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  journal={arXiv preprint arXiv:2410.05160},
  year={2024}
}

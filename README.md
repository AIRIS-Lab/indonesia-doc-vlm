# Indonesia-Doc-VLM

A collection of Vision-Language Models specialized for Indonesian document (KTP, KK, SIM, NPWP, etc.) data extraction.

[Screencast from 18-03-25 12:16:53.webm](https://github.com/user-attachments/assets/5830b26c-bbc1-425c-a767-1b96f2fe93f8)


## 🌟 Overview

Indonesia-Doc-VLM is an open-source project that provides specialized Vision-Language Models (VLMs) for extracting information from Indonesian documents. These models are fine-tuned to understand and extract data from various Indonesian identity and legal documents such as:

- KTP (Kartu Tanda Penduduk / Identity Card)
- KK (Kartu Keluarga / Family Card)
- SIM (Surat Izin Mengemudi / Driving License)
- NPWP (Nomor Pokok Wajib Pajak / Tax ID)
- And other Indonesian official documents

## 🏆 Available Models

The following models are available in this collection:

| Model | Size | Description |
|-------|------|-------------|
| [danielsyahputra/Qwen2-VL-2B-DocIndo](https://huggingface.co/danielsyahputra/Qwen2-VL-2B-DocIndo) | 2B | Base model for document understanding |
| [danielsyahputra/Qwen2-VL-2B-DocIndo-merged-16bit](https://huggingface.co/danielsyahputra/Qwen2-VL-2B-DocIndo-merged-16bit) | 2B | 16-bit quantized version for faster inference |
| [danielsyahputra/Qwen2.5-VL-3B-DocIndo](https://huggingface.co/danielsyahputra/Qwen2.5-VL-3B-DocIndo) | 3B | Larger model with improved accuracy |
| [danielsyahputra/Qwen2.5-VL-3B-DocIndo-merged-16bit](https://huggingface.co/danielsyahputra/Qwen2.5-VL-3B-DocIndo-merged-16bit) | 3B | 16-bit quantized version of the 3B model |
| [danielsyahputra/SmolVLM-500M-DocIndo](https://huggingface.co/danielsyahputra/SmolVLM-500M-DocIndo) | 500M | Lightweight model for resource-constrained environments |
| [danielsyahputra/SmolVLM-500M-DocIndo-merged-16bit](https://huggingface.co/danielsyahputra/SmolVLM-500M-DocIndo-merged-16bit) | 500M | 16-bit quantized version of the SmolVLM model |

## 🚀 Quick Start

### Installation

```bash
pip install unsloth transformers torch
```

### Using the models

```python
from unsloth import FastVisionModel
import torch

# Load the model (choose one from the available models)
model, tokenizer = FastVisionModel.from_pretrained(
    "danielsyahputra/Qwen2-VL-2B-DocIndo",
    load_in_4bit=True,  # For efficient memory usage
    use_gradient_checkpointing="unsloth",
    trust_remote_code=True,
)
FastVisionModel.for_inference(model)  # Enable for inference

# Load your image
from PIL import Image
image = Image.open("your_document.jpg")

# Define your prompt
instruction = "What are the NIK, nama, and alamat in the given image? Give me as a JSON"

# Prepare inputs for the model
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate output
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs, 
    streamer=text_streamer, 
    max_new_tokens=180,
    use_cache=True, 
    temperature=1.5, 
    min_p=0.1
)
```

## 📄 Supported Document Fields

The models can extract various fields from Indonesian documents, including but not limited to:

### KTP (Identity Card)
- NIK (ID Number)
- Nama (Name)
- Tempat/Tgl Lahir (Place/Date of Birth)
- Jenis Kelamin (Gender)
- Alamat (Address)
- RT/RW (Neighborhood units)
- Kel/Desa (Village)
- Kecamatan (District)
- Agama (Religion)
- Status Perkawinan (Marital Status)
- Pekerjaan (Occupation)
- Kewarganegaraan (Citizenship)

### NPWP (Tax ID)
- NPWP Number
- Name
- Address
- Registration Date

### SIM (Driving License)
- SIM Type
- SIM Number
- Name
- Address
- Validity Period

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the models or add support for additional document types, please feel free to submit a pull request.

## 📜 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact me on Hugging Face: [@danielsyahputra](https://huggingface.co/danielsyahputra)

- Email: **daniel.syahputra@akademiai.site**

## ⭐ Acknowledgements

- These models are fine-tuned versions of [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B) and [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- Thanks to the Hugging Face team for their platform and tools
- Special thanks to Unsloth for their optimization library

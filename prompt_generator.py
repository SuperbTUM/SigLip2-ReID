import random
from collections import defaultdict
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def sample_dataset(dataset, num_pids):
    samples = [None for _ in range(num_pids)]
    label2img = defaultdict(list)
    for data in dataset:
        img_path, pid = data[:2]
        label2img[pid].append(img_path)
    for pid in label2img:
        samples[pid] = random.choice(label2img[pid])
    return samples


def generate_prompt(image_paths, input_size):

    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.size = {"height": input_size[0], "width": input_size[1]}
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": img_path},
                    {"type": "text", "text": "Can you describe this image in one sentence?"},
                ]
            } 
        ] for img_path in image_paths
    ]

    all_generated_texts = []

    for messages in messages_batch:
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        # Generate text
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)

        # Decode output
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Collect the response
        all_generated_texts.append(generated_texts[0].split("\n")[1].replace("Assistant: ", ""))  # single response per image

    return all_generated_texts

def get_ai_prompt_by_dataset(dataset, num_pids, input_size):
    img_paths = sample_dataset(dataset, num_pids)
    return generate_prompt(img_paths, input_size)
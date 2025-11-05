import random
from PIL import Image
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
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a Re-Identification assistant. Focus on describing unique, identity-specific visual features, ignoring background or pose."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": img_path},
                    {"type": "text", "text": "Describe this image in one sentence."},
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

        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Collect the response
        all_generated_texts.append(generated_text.split("\n")[2].replace("Assistant: ", "").strip())  # single response per image

    return all_generated_texts

def get_ai_prompt_by_dataset(dataset, num_pids, input_size, dataset_name):
    img_paths = sample_dataset(dataset, num_pids)
    all_generated_texts = generate_prompt(img_paths, input_size)
    with open(f"prompts_{dataset_name}.txt", "w", encoding="utf-8") as f:
        for generated_prompt in all_generated_texts:
            f.write(generated_prompt + "\n")
    return all_generated_texts
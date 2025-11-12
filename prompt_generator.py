import re
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


def generate_prompt(image_paths, input_size, class_name):

    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
    processor.image_processor.size = {"height": input_size[0], "width": input_size[1]}
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    system_prompt = f"""
        You are a Re-Identification assistant for {class_name} object. Focus on describing unique, identity-specific visual features, ignoring background or pose of the {class_name} in the image. 
        Always start with "A {class_name}..." and keep the response in 16 words or fewer while providing as many notable details as possible.
    """
    user_prompts = [
        f"In one sentence, describe the unique visual traits that distinguish this {class_name} from others of the same type.",
        f"In one sentence, what fine-grained details identify this exact {class_name}?",
        f"In one sentence, summarize only the distinctive color, texture, and shape features of this {class_name}.",
    ]

    messages_batch = [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": img_path},
                    {"type": "text", "text": random.choice(user_prompts)},
                ]
            } 
        ] for img_path in image_paths
    ]

    all_generated_texts = []
    batch_size = 32

    for i in range(0, len(messages_batch), batch_size):
        # Apply chat template
        inputs = processor.apply_chat_template(
            messages_batch[i:i+batch_size],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(model.device, dtype=torch.bfloat16)

        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )

        # Decode
        for generated_text in processor.batch_decode(generated_ids, skip_special_tokens=True):
            # Collect the response
            generated_text = generated_text.split("\n")[5].replace("Assistant: ", "").strip()
            all_generated_texts.append(generated_text)  # single response per image

    return all_generated_texts

def get_ai_prompt_by_dataset(dataset, num_pids, input_size, dataset_name, class_name):
    img_paths = sample_dataset(dataset, num_pids)
    all_generated_texts = generate_prompt(img_paths, input_size, class_name)
    with open(f"prompts_{dataset_name}.txt", "w", encoding="utf-8") as f:
        for generated_prompt in all_generated_texts:
            f.write(generated_prompt + "\n")
    return all_generated_texts
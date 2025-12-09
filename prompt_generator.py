import random
from PIL import Image
from collections import defaultdict
import torch
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForImageTextToText

from constants import *

def sample_dataset(dataset, num_pids, input_size, vision_model):
    samples = [None for _ in range(num_pids)]
    label2img = defaultdict(list)
    for data in dataset:
        img_path, pid = data[:2]
        label2img[pid].append(img_path)
    for pid in label2img:
        samples[pid] = random.choice(label2img[pid])
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.5,
            std=0.5
        )
    ])
    batched_image = []
    for image_path in samples:
        image = Image.open(image_path).convert("RGB")
        tensor_image = transform(image)
        batched_image.append(tensor_image)
    batched_image = torch.stack(batched_image, dim=0)

    with torch.inference_mode():
        batched_feats = []
        for start_index in range(0, len(batched_image), BATCH_SIZE):
            feats = vision_model(batched_image[start_index:start_index+BATCH_SIZE].to(DEVICE), False)[0]
            feats = torch.nn.functional.normalize(feats)
            batched_feats.append(feats.half())
        batched_feats = torch.cat(batched_feats, dim=0)
    dist = 1 - batched_feats @ batched_feats.t()
    dist = dist + torch.eye(dist.size(0), device=dist.device) * 9999
    hardest_indices = dist.argmin(dim=1)
    hardest_samples = [samples[i] for i in hardest_indices.tolist()]
    return zip(samples, hardest_samples)


def generate_prompt(image_paths, input_size, class_name):

    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
    processor.image_processor.size = {"height": input_size[0], "width": input_size[1]}
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)

    system_prompt = f"""
        You are a Re-Identification assistant for {class_name} object. Focus on describing unique, identity-specific visual features, ignoring background or pose of the {class_name} in the image. 
        Your task is to describe only the identity-specific, unique visual features of Image A. Do not mention Image B directly. Only use it implicitly as a contrast. Start every response with: "The {class_name} in Image A".
        Keep the response in 16 words or fewer while providing as many notable details as possible.
    """
    user_prompts = [
        f"Describe the unique visual traits of Image A that distinguish this {class_name} from others. Only focus on the {class_name} appearance not motion.",
        f"What fine-grained details identify this exact {class_name} in Image A? Only focus on the {class_name} appearance not motion.",
        f"Summarize only the distinctive color, texture, and shape features of this {class_name} in Image A. Only focus on the {class_name} appearance not motion.",
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
                    {"type": "text", "text": "Image A above."},
                    {"type": "image", "path": hard_sample},
                    {"type": "text", "text": "Image B above."},
                    {"type": "text", "text": random.choice(user_prompts)},
                ]
            } 
        ] for img_path, hard_sample in image_paths
    ]

    all_generated_texts = []
    batch_size = 16

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
                max_new_tokens=21,
                do_sample=False,
            )

        # Decode
        for generated_text in processor.batch_decode(generated_ids, skip_special_tokens=True):
            # Collect the response
            generated_text = generated_text.split("\n")[6].replace("Assistant: ", "").replace(f"The {class_name} in Image A", "").strip()
            all_generated_texts.append(generated_text)  # single response per image

    return all_generated_texts

def get_ai_prompt_by_dataset(dataset, num_pids, input_size, dataset_name, class_name, vision_model):
    img_paths = sample_dataset(dataset, num_pids, input_size, vision_model)
    all_generated_texts = generate_prompt(img_paths, input_size, class_name)
    with open(f"prompts_{dataset_name}.txt", "w", encoding="utf-8") as f:
        for generated_prompt in all_generated_texts:
            f.write(generated_prompt + "\n")
    return all_generated_texts
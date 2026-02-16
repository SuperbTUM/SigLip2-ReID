import glob
import base64
import random
import json
from collections import defaultdict

from dotenv import load_dotenv
import os
load_dotenv()
# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# make it a batch request
def generate_batch_requests(base_path, class_name, dataset, input_size):
    # Path to your image
    image_paths = glob.glob("/".join([base_path, "*.jpg"]))

    image_list = defaultdict(list)

    for image_path in image_paths:
        image_name = image_path.split("/")[-1]
        label = int(image_name.split("_")[0])
        image_list[label].append(image_path)

    assert len(image_list) < 1000
    if class_name == "person":
        user_prompt = f"Keep image with size of {input_size}. Focus on the {class_name} in the photos. Summarize the common parts of the {class_name}'s top, bottom and shoes and avoid camera angle, viewpoint and behavior in one sentence and under 30 words starting with 'A photo of a'."
    else:
        user_prompt = f"Keep image with size of {input_size}. Focus on the {class_name} in the photos. Summarize the common parts of the {class_name}'s appearance and avoid camera angle, viewpoint and behavior in one sentence and under 30 words starting with 'A photo of a'."

    with open(f"requests_{dataset}_full.jsonl", "w+") as f:
        for label in image_list:

            model = "gpt-5-mini"

            for attempt in range(2):

                image_path, image_path2 = random.sample(image_list[label], 2)

                # Getting the base64 string
                base64_image = encode_image(image_path)
                base64_image2 = encode_image(image_path2)

                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image2}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_completion_tokens": 1536
                }

                request = {"custom_id": "_".join((str(label), str(attempt))),"method": "POST", "url": "/v1/chat/completions", "body": payload}


                f.write(json.dumps(request) + "\n")

'''
# Upload the file
# curl https://api.openai.com/v1/files \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -F "purpose=batch" \
#   -F "file=@requests.jsonl"
# file-8eb9i95KBbZUB2ZUW5SFAK

# Create batch job
curl https://api.openai.com/v1/batches \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input_file_id": "file-8eb9i95KBbZUB2ZUW5SFAK", "endpoint": "/v1/chat/completions", "completion_window": "24h"}'
# batch_69576b969f0c81908561ab8c421bb086
  
# Check batch status
curl https://api.openai.com/v1/batches/batch_69576b969f0c81908561ab8c421bb086 \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Download batch result
curl https://api.openai.com/v1/files/file-JdeqVBx6MFX3gxXgkkfcSw/content \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -o results.jsonl
'''

def generate_descriptions(result_path, dataset):
    rows = []
    with open(result_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            custom_id = int(raw["custom_id"].split("_")[0])
            response = raw["response"]["body"]["choices"][0]["message"]["content"]
            rows.append((custom_id, response))
    rows.sort(key=lambda x: x[0])
    with open(f"prompts_{dataset}_full.txt", "w+") as f:
        for cid, content in rows:
            if not content:
                content = "[EMPTY]"
            f.write(f"{content}\n")

if __name__ == "__main__":
    generate_batch_requests("Market-1501-v15.09.15/bounding_box_train/", "person", "market", (256, 128))
    generate_batch_requests("VeRi/image_train/", "vehicle", "veri", (224, 224))
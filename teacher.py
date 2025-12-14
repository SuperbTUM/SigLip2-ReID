import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from constants import DEVICE

device = DEVICE
teacher_model_name = "Salesforce/blip2-opt-2.7b"

processor = Blip2Processor.from_pretrained(
    teacher_model_name
)

model = Blip2ForConditionalGeneration.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

@torch.no_grad()
def blip2_scores_batch(images, texts):
    """
    images: torch.Tensor (B,3,H,W) in [0,1]
    texts:  List[str] length B
    """
    inputs = processor(
        images=images,
        text=texts,
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device, torch.float16) for k, v in inputs.items()}

    outputs = model(**inputs)

    logits = outputs.logits  # (B, T, V)
    labels = inputs["input_ids"]  # (B, T)

    # Shift for causal LM
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none"
    )

    loss = loss.view(shift_labels.size())  # (B, T-1)

    # Mask padding
    mask = shift_labels != processor.tokenizer.pad_token_id
    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)

    scores = -loss  # higher = better
    return scores

def teacher_distribution(scores, temperature=1.0):
    """
    scores: Tensor (N_candidates,) or (B, N_candidates)
    """
    if scores.ndim == 1:
        return F.softmax(scores / temperature, dim=0)
    else:  # batch of images
        return F.softmax(scores / temperature, dim=1)

def denormalize(img):
    return img * torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(img.device) + torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1).to(img.device)

def teacher_model_output(dataloaders, texts):
    dists_by_dataset = []
    for i, train_dataloader in enumerate(dataloaders):
        dists = []
        for batch in train_dataloader:
            image_tensor, label = batch[:2]
            image_tensor = image_tensor.to(device)
            label = label.to(device)
            text = texts[i][label]
            dist = teacher_distribution(blip2_scores_batch(denormalize(image_tensor), text))
            dists.append(dist)
        dists_by_dataset.append(torch.cat(dists, dim=0))
    return dists_by_dataset

def teacher_student_loss(teacher_dist, student_dist, eps=1e-8):
    return F.kl_div((student_dist + eps).log(), teacher_dist, reduction="batchmean")
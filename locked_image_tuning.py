import model
from constants import *

import transformers
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0

    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss
    
class PromptLearner(nn.Module):
    def __init__(self, num_prompt_tokens, embedding_dim, class_names):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prompt_tokens = num_prompt_tokens

        # Initialize the learnable prompt vectors
        prompt_vectors = torch.empty(len(class_names), num_prompt_tokens, embedding_dim)
        nn.init.normal_(prompt_vectors, std=0.02) # Standard initialization
        self.prompt = nn.Parameter(prompt_vectors)

        # Store tokenized class names
        self.class_names = class_names
        # Note: In a full implementation, you'd handle tokenization carefully here.

    def forward(self, text_encoder, text_tokenizer):
        # Tokenize the class names
        text_inputs = text_tokenizer(self.class_names, padding=True, return_tensors="pt", return_attention_mask=True)
        
        # Get the standard word embeddings for class names
        class_name_embs = text_encoder(text_inputs["input_ids"].to(self.prompt.device), text_inputs["attention_mask"].to(self.prompt.device),
                                       use_pooled_output=False)
        
        # Get the learnable prompt
        learnable_prompt = self.prompt
        
        # Prepend the learnable prompt to the class name embeddings
        # [PROMPT, PROMPT, ..., CLASS_NAME]
        combined_embs = torch.cat([learnable_prompt, class_name_embs], dim=1)
        
        # Pass the combined embeddings through the rest of the text encoder
        # This part requires a custom forward pass through the text model layers
        # For simplicity, we assume we can pass embeddings directly. 
        # In transformers, you pass it to the encoder layers.
        
        # A simplified representation of passing through the encoder:
        # Note: The actual `transformers` implementation requires passing embeddings
        # through `model.text_model.encoder` and then `model.text_model.final_layer_norm`.
        prompted_text_features = text_encoder(inputs_embeds=combined_embs)
        
        return prompted_text_features

def train_one_pair(text, image_tensor, text_label, image_label, device):
    """
    A quick example of training sigLip2 with LiT
    """
    tokenizer = transformers.Autotokenizer.from_pretrained(MODEL_NAME)
    textual_inputs = tokenizer(text, padding=True, return_tensor="pt", return_attention_mask=True)
    vl_model = model.load_weights(MODEL_NAME, False)
    # prompt_learner = PromptLearner(4, 768, ...)
    # Load optimizer
    optimizer = torch.optim.Adam(vl_model.text_model.parameters(), lr=0.001, weight_decay=1e-4)

    text_features = vl_model.get_text_features(input_ids=textual_inputs["input_ids"], attention_mask=textual_inputs["attention_masks"])
    image_features = vl_model.get_image_features(image_tensor) # This can be pre-loaded
    optimizer.zero_grad()
    sup_con_loss = SupConLoss(device)
    optimizer.step()
    return sup_con_loss(text_features, image_features, text_label, image_label) + \
            sup_con_loss(image_features, text_features, image_label, text_label)

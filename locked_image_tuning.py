import model
from constants import *

import transformers
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig


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


def LoRA_tuning_one_batch(image_label, text_label, device):
    # --- Setup from before ---
    base_model = model.load_weights(MODEL_NAME).to(device)
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )
    base_model.text_model = get_peft_model(base_model.text_model, lora_config)
    lora_model = base_model
    # --- End of previous setup ---


    # 1. Instantiate the Prompt Learner
    # We need to know the embedding dimension of our text model.
    embedding_dim = lora_model.config.text_config.hidden_size
    prompt_learner = PromptLearner(
        num_prompt_tokens=4, # hyper-parameter
        embedding_dim=embedding_dim,
        class_names=["example_class_" + str(i) for i in range(N_CLS)]
    ).to(device)

    # 2. Define the Optimizer to train BOTH sets of parameters
    # This is the crucial step: you combine the parameters from both the
    # prompt_learner and the LoRA-adapted model.
    trainable_params = list(prompt_learner.parameters()) + list(lora_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=5e-4, weight_decay=1e-4)

    # --- Conceptual Training Step ---

    # Dummy inputs
    images = torch.rand(2, 3, 224, 224).to(device) # Dummy images

    # 3. Use the Prompt Learner
    # Pass the initial embeddings through the prompt learner to get the combined embeddings.
    modified_text_embeddings = prompt_learner(lora_model.text_model, text_tokenizer)

    # 4. Forward pass through the LoRA-adapted model
    # The model now receives the modified input.
    # Note: This is a simplified forward pass. The actual call might differ based on the model.
    # The key is that `modified_text_embeddings` is the input, not the original `token_embeddings`.
    text_features = modified_text_embeddings[text_label]
    image_features = lora_model.get_image_features(images)

    # Your loss calculation and backpropagation would follow...
    optimizer.zero_grad()
    sup_con_loss = SupConLoss(device)
    loss = sup_con_loss(text_features, image_features, text_label, image_label) + \
            sup_con_loss(image_features, text_features, image_label, text_label)
    loss.backward()
    optimizer.step()
    return loss.item()

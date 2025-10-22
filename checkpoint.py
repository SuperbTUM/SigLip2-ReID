import torch

def save_checkpoint(
        text_model,
        prompt_learners,
        epoch
):
    path = f"checkpoint_epoch_{epoch}.pth"
    prompter_states = [p.state_dict() for p in prompt_learners]
    # Create a dictionary with all the states
    checkpoint = {
        'epoch': epoch,
        'text_model_state_dict': text_model.state_dict(),
        'prompt_learners_state_dict': prompter_states,
    }

    # Save the checkpoint dictionary
    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")


def load_checkpoint(
        model,
        prompt_learners,
        path,
        device):
    checkpoint = torch.load(path, map_location=device)
    model.text_model.load_state_dict(checkpoint["text_model_state_dict"])
    loaded_prompter_states = checkpoint["prompter_learners_state_dict"]
    for i, state_dict in enumerate(loaded_prompter_states):
        prompt_learners[i].load_state_dict(state_dict)
    return model, prompt_learners

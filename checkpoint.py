import torch

def save_checkpoint(
        model,
        prompt_learners,
        temperature,
        epoch,
        classifiers=None,
        optimizer=None,
        scheduler=None
):
    path = f"checkpoint_epoch_{epoch}.pth"
    prompter_states = [p.state_dict() for p in prompt_learners] if prompt_learners is not None else []
    classifiers_states = [c.state_dict() for c in classifiers] if classifiers is not None else []
    # Create a dictionary with all the states
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'prompt_learners_state_dict': prompter_states,
        'classifiers_state_dict': classifiers_states,
        'temperature': temperature,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    # Save the checkpoint dictionary
    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")


def load_checkpoint(
        model,
        prompt_learners,
        classifiers,
        optimizer,
        scheduler,
        path,
        device):
    checkpoint = torch.load(path, map_location=device)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if prompt_learners is not None:
        loaded_prompter_states = checkpoint["prompt_learners_state_dict"]
        for i, state_dict in enumerate(loaded_prompter_states):
            prompt_learners[i].load_state_dict(state_dict)
    if classifiers is not None:
        load_classifier_states = checkpoint['classifiers_state_dict']
        for i, state_dict in enumerate(load_classifier_states):
            classifiers[i].load_state_dict(state_dict)
    temperature = checkpoint["temperature"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return model, prompt_learners, classifiers, temperature, optimizer, scheduler

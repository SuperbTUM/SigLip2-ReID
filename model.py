import transformers
from siglip import SiglipvisionConfig, SiglipTextConfig, SiglipConfig, SiglipModel



def load_weights(model_name, text_model_loaded=True):
    model = transformers.SiglipModel.from_pretrained(model_name)

    model_dict = model.state_dict()
    vision_model_dict = model.vision_model.state_dict()

    vision_config = SiglipvisionConfig(version="siglip2" if "siglip2" in model_name else "siglip")
    text_config = SiglipTextConfig()
    siglip_config = SiglipConfig(text_config=text_config.__dict__, vision_config=vision_config.__dict__)
    reproduced_model = SiglipModel(siglip_config)
    if text_model_loaded:
        reproduced_model.load_state_dict(model_dict)
    else:
        reproduced_model.vision_model.load_state_dict(vision_model_dict)
    reproduced_model.eval()
    return reproduced_model


def get_model_outputs(texts_input, images_input, model, model_layout_name):
    interpolate_pos_encoding = model_layout_name[-3:].isdigit() and (images_input.size(2) == int(model_layout_name[-3:]) and images_input.size(3) == int(model_layout_name[-3:]))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_layout_name)
    tokenized_inputs = tokenizer(texts_input, padding=True, return_tensors="pt", return_attention_mask=True)
    return model(input_ids=tokenized_inputs["input_ids"], attention_mask=tokenized_inputs["attention_mask"],
                 pixel_values=images_input, interpolate_pos_encoding=interpolate_pos_encoding)

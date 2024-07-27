def get_model(PreTrainedConfig, model_name):
    return PreTrainedConfig.from_pretrained(model_name)
    
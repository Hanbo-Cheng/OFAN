import torch 

def load_encoder(model, offline_path, online_path):
    print("load from : ", offline_path)
    model_dict = model.state_dict()
    off_params = torch.load(offline_path)
    for k in off_params.keys():
        if('encoder' in k):
            k_t = k.split('.')
            k_t[0] = 'encoder_off'
            k_t = '.'.join(k_t)
            model_dict[k_t] = off_params[k]
    
    print("load from : ", online_path)
    on_params = torch.load(online_path)
    for k in on_params.keys():
        if('encoder' in k):
            k_t = k.split('.')
            k_t[0] = 'encoder_on'
            k_t = '.'.join(k_t)
            model_dict[k_t] = on_params[k]
    model.load_state_dict(model_dict)

def load_off_encoder(model, offline_path, online_path):
    model_dict = model.state_dict()
    off_params = torch.load(offline_path)
    for k in off_params.keys():
        if('encoder' in k):
            model_dict[k] = off_params[k]
    model.load_state_dict(model_dict)


def load_all(model, offline_path, online_path):
    model_dict = model.state_dict()
    off_params = torch.load(offline_path)
    for k in off_params.keys():
        
        model_dict[k] = off_params[k]
    model.load_state_dict(model_dict)

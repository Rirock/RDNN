import os
import shutil

def format_runtime(time_gap):
    m, s = divmod(time_gap, 60)
    h, m = divmod(m, 60)
    runtime_str = ''
    if h != 0:
        runtime_str = runtime_str + '{}h'.format(int(h))
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    elif m != 0:
        runtime_str = runtime_str + '{}m'.format(int(m))
        runtime_str = runtime_str + '{}s'.format(int(s))
    else:
        runtime_str = runtime_str + '{:.4}s'.format(s)
    return runtime_str

def copy_codes(src_path, dst_path, files = ["main.py", "train.py", "network/", "config.json"]):
    cp_path = os.path.join(dst_path, "codes")
    if not os.path.exists(cp_path):
        os.mkdir(cp_path)
    for file in files:
        if "/" in file:
            last_slash_index = file.rfind("/")
            if last_slash_index != -1:    
                folder_name = file[:last_slash_index+1]
            else:     
                folder_name = file
            if not os.path.exists(os.path.join(cp_path, folder_name)):
                os.makedirs(os.path.join(cp_path, folder_name))
        old_path = os.path.join(src_path, file)
        new_path = os.path.join(cp_path, file)
        if os.path.isdir(old_path):
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
            shutil.copytree(old_path, new_path)
        else:
            shutil.copyfile(old_path, new_path)

def transform_xy(model_name, x, y, device="cpu"):
    if "RDNN" in model_name or "REG" in model_name:
        x = x.transpose(0, 1).to(device).float()
        y = y.transpose(0, 1).to(device).float()
    elif "MLP" in model_name:
        x = x.reshape(x.size(0), -1).to(device).float()
        y = y.reshape(y.size(0), -1).to(device).float()

    return x, y

def transform_outy(model_name, out, y):
    if "MLP" in model_name:
        y = y.transpose(0, 1).float()
        out = out.transpose(0, 1).float()
        
    y = y[-1,:].cpu().detach().numpy()
    out = out[-1,:].cpu().detach().numpy()

    return out, y
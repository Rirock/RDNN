import os
import shutil
import time
import json

import platform


# get the current timestamp
current_timestamp = time.time()
local_time = time.localtime(current_timestamp)

data_path = "./data"
file_names = os.listdir(data_path)
model_names = ["RDNN", "LSTM_REG", "GRU_REG", "MLP"]
model_name = model_names[0]

M_list = range(2, 21, 2)
M = 10
run_times = 10
use_M = False

if platform.system().lower() == 'windows':
    ext = ".bat"
elif platform.system().lower() == 'linux':
    ext = ".sh"

main_py = "main"  # "main_s"

log_name = time.strftime("%Y-%m-%d-%H", local_time)

# load config
config_file = open('config.json', 'r').read()
config = json.loads(config_file)
baseDir = os.path.dirname(os.path.abspath(__name__))
log_root = os.path.join(baseDir, config['log_params']['log_root']+"_"+model_name+"_"+log_name)

if not os.path.exists(log_root):
    os.makedirs(log_root)

sh_name = "run_"+model_name+ext
sh_save_path = os.path.join(log_root, sh_name)

f = open(sh_name, "w", encoding="utf-8")

if use_M:
    for M in M_list:
        code = 'python '+main_py+'.py -m "'+model_name+'" -d '+'"'+data_path+'"'+' -n '+str(run_times)+' --DNM_M '+str(M)+' -l '+log_name+'\n'
        f.write(code)
        print(code)
else:
    code = 'python '+main_py+'.py -m "'+model_name+'" -d '+'"'+data_path+'"'+' -n '+str(run_times)+' --DNM_M '+str(M)+' -l '+log_name+'\n'
    f.write(code)
    print(code)
f.close()

shutil.copy(sh_name, sh_save_path)
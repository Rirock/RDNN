import os
import csv
import numpy as np
import scipy.stats as stats

import os
import shutil


def cp_logs_by_model(new_log_path, floder_name, data_names):
    # copy logs by model name

    if not os.path.exists(floder_name):
        print("floder name not exist!")
        return 
    
    if not os.path.exists(new_log_path):
        os.mkdir(new_log_path)

    for data_name in data_names:
        # data_name = data_name[:-4]
        old_path = os.path.join(floder_name, "logs_test", data_name)
        new_path = os.path.join(new_log_path, data_name)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        files = os.listdir(old_path)

        for file in files:
            old_path_f = os.path.join(old_path, file)
            new_path_f = os.path.join(new_path, file)
            shutil.copyfile(old_path_f, new_path_f)


if __name__ == "__main__":
    
    # naming of each model
    models = ["RDNN_M2", "MLP", "LSTM_REG", "GRU_REG"]
    # the path where each model log is located
    floder_names = ["logs_RDNN_2023-05-24-18", "logs_MLP_2023-05-24-10", "logs_LSTM_REG_2023-05-23-12", "logs_GRU_REG_2023-05-18-21"]

    # dataset for testing
    data_names = ["N225", "IXIC", "AXJO", "FCHI", "TASI", "IMOEX", "BVSP", "DAX30", "SSEC", "SPX", "KS11", "SZI", "NSEI", "DJI"]

    new_log_path = "./logs_result"
    if os.path.exists(new_log_path):
        shutil.rmtree(new_log_path)

    for floder_name in floder_names:
        cp_logs_by_model(new_log_path, floder_name, data_names)

    nameEnd = "_pred.csv"

    folder_paths = []
    for folder in data_names:
        folder_path = os.path.join(new_log_path, folder)
        if os.path.isdir(folder_path):
            folder_paths.append(folder)

    result = []
    jsnum = [0 for _ in range(len(models))]

    w_list = np.zeros(len(models))
    t_list = np.zeros(len(models))
    l_list = np.zeros(len(models))

    metrics_i = 0

    for file_name in folder_paths:
        logDir = os.path.join(new_log_path, file_name.split(".")[0])
        print(file_name.split(".")[0])
        r_m = []
        
        for m_i, model in enumerate(models):
            logDiri = os.path.join(logDir, model+nameEnd)
            r = []
            csv_r = csv.reader(open(logDiri, "r"))
            for i, num in enumerate(csv_r):
                num = [float(item) for item in num]
                r.append(num)

            # Delete data with r2 less than 0
            # r = [x for x in r if x[-1]>0]
            # for i, rr in enumerate(r):
            #     if float(rr[-1]) < 0:
            #         for j in range(len(r)):
            #             del r[j][i]

            r = np.array(r)
            r_mean = np.mean(r, axis=0)
            print("{:10}: \t{:.3f}   \t {:.3f}   \t {:.3f}   \t {:.3f}".format(model, r_mean[0], r_mean[1], r_mean[2], r_mean[3]))

            if "RDNN" in model:
                r_dnm = r
            else:
                stat, pAB = stats.ranksums(r_dnm[:, metrics_i], r[:, metrics_i], alternative='less')
                stat, pBA = stats.ranksums(r[:, metrics_i], r_dnm[:, metrics_i], alternative='less')

                if pAB <= 0.05 and pBA >= 0.05:
                    w_list[m_i] += 1
                elif pAB >=0.05 and pBA <= 0.05:
                    l_list[m_i] += 1
                else:
                    t_list[m_i] += 1

            r_m.append(r_mean)

        n = np.argmin(r_m, axis=0)[0]
        jsnum[n] += 1
        print(models[n])
        print()
        result.append(r_m)

    print("Optimal number: ", jsnum)
    print()
    print("Win: ", w_list)
    print("Tie: ", t_list)
    print("Lose:", l_list)

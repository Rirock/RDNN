
##  main.py
### Usage examples:
python main.py -m "RDNN" -d "./data5" -n 10 --DNM_M 2 -l 2023-05-21-15

### Parameter Description:
python main.py 

- -m: model_name 
- -d: data_path
- -n: run times 
- --DNM_M: DNM Hyperparameter M
- -l: The extension of the log path (This is just a marker where you can enter any character you like)

----
##  get_command.py
Get the script to run the code. You can try modifying the following parts of the code.
```
model_names = ["RDNN", "LSTM_REG", "GRU_REG", "MLP"]
model_name = model_names[0]  # Generate a run script for the first model "RDNN"

M = 10
run_times = 10
```
After runing *"python get_command.py"*, you will get a file "run_RDNN.sh" or "run_RDNN.bat".

When your environment is Windows, double-click the "run_RDNN.bat" file directly to run it. If the environment is Linux, please use "sh run_RDNN.sh" to run the script.

----
##  config.json
This is where you set the data, model, run and other parameters. Before you start experimenting, please modify this file according to your own dataset.

### "data_params": 
- "time_step": How many days of data to use as input
- "predict_day": How many days of data to predict
- "features":  ["S":Single Sequence, "MS":Multi-Sequence -> Single Sequence, "M":Multi-Sequence]
- "cols": Input Sequence. Such as: ["Close"]
- "target": Outnput Sequence

----
##  get_result.py
Run *"python get_result.py"* to get the results of the experiment.

You need to make changes to the following two sections:
### models: which model results to output
```
models = ["RDNN_M2", "MLP", "LSTM_REG", "GRU_REG"]
```
### floder_names: the path of logs of different models
```
floder_names = ["logs_RDNN_2023-05-24-18", "logs_MLP_2023-05-24-10", "logs_LSTM_REG_2023-05-23-12", "logs_GRU_REG_2023-05-18-21"]
```

----
## ./network/DNM_models.py
The performance of RDNN may not work well on some datasets and changes can be made to the structure of the model here.

----
## Some code that does not need to be changed in general：

- dataloader.py:

The code used to read the dataset, you normally don't need to make changes here. If an error occurs, please check the dataset with the config settings first.

- train.py 
- predict.py
- utils.py


----
----
# 参考までに、ChatGPTの翻訳結果を紹介します。


## main.py
#### 使用例：
python main.py -m "RDNN" -d "./data5" -n 10 --DNM_M 2 -l 2023-05-21-15

#### パラメータの説明：
python main.py

- -m: モデル名
- -d: データパス
- -n: 実行回数
- --DNM_M: DNMハイパーパラメータM
- -l: ログパスの拡張子（好きな文字を入力できるマーカーです）

----
## get_command.py
コードを実行するためのスクリプトを取得します。コードの以下の部分を変更してみることができます。

```
model_names = ["RDNN", "LSTM_REG", "GRU_REG", "MLP"]
model_name = model_names[0]  # 最初のモデル "RDNN" の実行スクリプトを生成します

M = 10
run_times = 10
```
"python get_command.py" を実行すると、"run_RDNN.sh" または "run_RDNN.bat" というファイルが生成されます。

環境がWindowsの場合、"run_RDNN.bat" ファイルをダブルクリックして直接実行します。環境がLinuxの場合は、スクリプトを実行するために "sh run_RDNN.sh" を使用してください。

----
## config.json
ここでデータ、モデル、実行などのパラメータを設定します。実験を開始する前に、独自のデータセットに基づいてこのファイルを編集してください。

- "data_params":
- "time_step": 入力として使用する日数の数
- "predict_day": 予測する日数の数
- "features": ["S":単一の系列, "MS":複数の系列 -> 単一の系列, "M":複数の系列]
- "cols": 入力系列。例: ["Close"]
- "target": 出力系列

----
## get_result.py
"python get_result.py" を実行すると、実験の結果を取得できます。

以下の2つのセクションを変更する必要があります。
#### models: 出力するモデルの結果
```
models = ["RDNN_M2", "MLP", "LSTM_REG", "GRU_REG"]
```
#### floder_names: 異なるモデルのログのパス
```
floder_names = ["logs_RDNN_2023-05-24-18", "logs_MLP_2023-05-24-10", "logs_LSTM_REG_2023-05-23-12", "logs_GRU_REG_2023-05-18-21"]
```

----
## ./network/DNM_models.py
RDNNのパフォーマンスが一部のデータセットでうまく機能しない場合、モデルの構造を変更することができます。

----
## 一般的に変更する必要のないコード：
- dataloader.py:
データセットを読み込むために使用されるコードで、通常ここでは変更する必要はありません。エラーが発生した場合は、まず構成設定でデータセットを確認してください。

- train.py
- predict.py
- utils.py
# Fed


## Requirements
- python 3
- see `requirements.txt`


# step 1: Data Preparation

mkdir -p data/{PEMS08, PEMS04, PEMS228, PEMS-BAY}

### Download PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).
### Download PEMS03，PEMS04， PEMS07 and PEMS08 datasets provided by [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). 

Rename downloaded PEMSXX。npz to pemsXX.npz and put the pems03.npz, pems04.npz, pems07.npz, pems08.npz, pems-bay.h5, metr-la.h5 to the directory data/{PEMS03, PEMS04, PEMS07, PEMS08, PEMS-BAY, METR-LA} respectively.



# Step2: Process raw data 

```

# PEMS08
python Preprocess_Data.py.py --output_dir=data/PEMS08 --traffic_df_filename=data/PEMS08/pems08.npz --data=PEMS08

# PEMS04
python Preprocess_Data.py.py --output_dir=data/PEMS04 --traffic_df_filename=data/PEMS04/pems04.npz --data=PEMS04

# PEMS03
python Preprocess_Data.py.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/PEMS-ABY/pems-bay.h5 --data=PEMS-BAY

# PEMS07
python Preprocess_Data.py.py --output_dir=data/PEMS228 --traffic_df_filename=data/PEMS228/V_228.csv --data=PEMS228

```
# step 3: Train


* PEMS04 dataset
```
python train.py --adjtype=doubletransition --adjdata=data/sensor_graph/dis_adj_index_04.csv --data=data/PEMS04 --num_node=307
```

* PEMS08 dataset
``` 
python train.py --adjtype=doubletransition --adjdata=data/sensor_graph/dis_adj_index_08.csv --data=data/PEMS08  --num_node=170
```

* PEMS-BAY dataset
```
python train.py --adjtype=doubletransition --adjdata=data/sensor_graph/pems_bay_adj.csv --data=data/PEMS-BAY  --num_node=325
```

* PEMS228 dataset
```
python train.py --adjtype=doubletransition --adjdata=data/sensor_graph/dis_W_228.csv --data=data/PEMS228  --num_node=228
```

# step 3: Test
xxxx correspond to the saved model file

* PEMS04 dataset
```
python test.py --adjtype=doubletransition --adjdata=data/sensor_graph/dis_adj_index_04.csv --data=data/PEMS04 --num_node=307 --checkpoint=xxxx
```

* PEMS08 dataset
``` 
python test.py --adjtype=doubletransition --adjdata=data/sensor_graph/dis_adj_index_08.csv --data=data/PEMS08  --num_node=170 --checkpoint=xxxx
```

* PEMS03 dataset
```
python test.py --adjtype=doubletransition --adjdata=data/sensor_graph/pems_bay_adj.csv --data=data/PEMS-BAY  --num_node=325 --checkpoint=xxxx
```

* PEMS07 dataset
```
python test.py --adjtype=doubletransition --adjdata=data/sensor_graph/dis_W_228.csv --data=data/PEMS228  --num_node=228 --checkpoint=xxxx
```



data:
python Preprocess_Data.py

Run：
python train.py

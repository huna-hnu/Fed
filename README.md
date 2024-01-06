# Fed


## Requirements
- python 3
- pytoch>=1.10


# step 1: Data Preparation

mkdir -p data/{PEMS03, PEMS04,PEMS07,PEMS08,  PEMS-BAY, METR-LA}

### Download PEMS-BAY and METR-LA data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).
### Download PEMS03，PEMS04， PEMS07 and PEMS08 datasets provided by [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). 

Rename downloaded PEMSXX.npz to pemsXX.npz and put the pems03.npz, pems04.npz, pems07.npz, pems08.npz, pems-bay.h5, metr-la.h5 to the directory data/{PEMS03, PEMS04, PEMS07, PEMS08, PEMS-BAY, METR-LA} respectively.



# Step2: Process raw data 

```

# PEMS08
python Preprocess_Data.py.py --dataset_name=PEMS08 --graph_signal_matrix_filename=data/PEMS08/pems08.npz --num_of_vertices=170

# PEMS04
python Preprocess_Data.py.py  --dataset_name=PEMS04 --graph_signal_matrix_filename=data/PEMS04/pems04.npz --num_of_vertices=307

# PEMS03
python Preprocess_Data.py.py  --dataset_name=PEMS03 --graph_signal_matrix_filename=data/PEMS03/pems03.npz --num_of_vertices=358

# PEMS07
python Preprocess_Data.py.py  --dataset_name=PEMS07 --graph_signal_matrix_filename=data/PEMS07/pems07.npz --num_of_vertices=883

# PEMS-BAY
python Preprocess_Data.py.py  --dataset_name=PEMS-BAY --graph_signal_matrix_filename=data/PEMS-BAY/pems-bay.h5 --num_of_vertices=358

# METR-LA
python Preprocess_Data.py.py  --dataset_name=METR-LA --graph_signal_matrix_filename=data/METR-LA/metr-la.h5 --num_of_vertices=207

```
# step 3: Run

* PEMS03 dataset
```
python train.py --dataset=PEMS03 --datafile=data/PEMS03/pems03.npz 
```

* PEMS04 dataset
```
python train.py --dataset=PEMS04 --datafile=data/PEMS04/pems04.npz 
```

* PEMS07 dataset
```
python train.py --dataset=PEMS07 --datafile=data/PEMS07/pems07.npz 
```

* PEMS08 dataset
``` 
python train.py --dataset=PEMS08 --datafile=data/PEMS08/pems08.npz 
```

* PEMS-BAY dataset
```
python train.py --dataset=PEMS-BAY --datafile=data/PEMS-BAY/pems-bay.h5
```

* METR-LA dataset
```
python train.py --dataset=METR-LA --datafile=data/METR-LA/metr-la.h5
```


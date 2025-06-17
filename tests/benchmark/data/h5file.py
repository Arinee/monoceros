import h5py
import pandas as pd
import os

def hdf5_to_csv(hdf5_file, csv_file_prefix):
    # 打开 HDF5 文件
    with h5py.File(hdf5_file, 'r') as hdf:
        # 遍历文件中的所有数据集
        for dataset_name in hdf.keys():
            # 读取数据集
            data = hdf[dataset_name][:]
            
            # 将数据集转换为 DataFrame
            df = pd.DataFrame(data)
            
            # 将 DataFrame 写入 CSV 文件
            csv_file = f"{csv_file_prefix}_{dataset_name}.csv"
            df.to_csv(csv_file, index=False)

def csv_to_hdf5(csv_files, hdf5_file):
    # 打开 HDF5 文件以便写入
    with h5py.File(hdf5_file, 'w') as hdf:
        # 遍历所有 CSV 文件
        for csv_file in csv_files:
            # 获取数据集名称（去掉文件的目录和扩展名）
            dataset_name = (os.path.splitext(os.path.basename(csv_file))[0]).split("_")[1]
            
            # 读取 CSV 文件到 DataFrame
            df = pd.read_csv(csv_file)
            print(df)
            
            # 将 DataFrame 转换为 NumPy 数组
            data = df.to_numpy()
            
            # 将数据写入 HDF5 文件中的数据集
            hdf.create_dataset(dataset_name, data=data)

if __name__ == "__main__":
    # hdf5转成csv
    hdf5_file = 'sift-128-euclidean.hdf5'  # 替换为你的 HDF5 文件名
    csv_file_prefix = 'sift'  # CSV 文件名前缀
    hdf5_to_csv(hdf5_file, csv_file_prefix)

    # csv转成hdf5
    # 列出要组合的 CSV 文件
    # csv_files = [
    #     'dssmtopk_train.csv',
    #     'dssmtopk_test.csv',
    #     'dssmtopk_neighbors.csv',
    #     'dssmtopk_distances.csv'
    # ]
    # hdf5_file = 'dssmtopk-64-innerproduct.hdf5'

    # csv_to_hdf5(csv_files, hdf5_file)
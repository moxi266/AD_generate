import numpy.core.multiarray
from monai.data import MetaTensor
from torch.serialization import add_safe_globals

# 允许所有必要的类
add_safe_globals([
    numpy.core.multiarray._reconstruct,  # 允许 numpy 数组
    MetaTensor,                          # 允许 MONAI 的 MetaTensor
    # 如果报错涉及其他类（如其他 MONAI 类型），继续添加
])
import os
from typing import Optional, Union

import torch
import pandas as pd
from monai.data import Dataset, PersistentDataset
from monai.transforms.transform import Transform


class DebugPersistentDataset(PersistentDataset):
    def __getitem__(self, index):
        try:
            # 调用父类 __getitem__ 获取样本数据
            sample = super().__getitem__(index)
            return sample
        except Exception as e:
            # 假设样本中有 'image_path' 字段记录文件路径
            # 如果您的键名称不同，请修改这里
            try:
                pet_path = self.data[index]['pet']
                mri_path = self.data[index]['mri']
            except Exception:
                pet_path = f"Index {index} (path not available)"
                mri_path = f"Index {index} (path not available)"

            print(f"Error processing sample at index {index}, path: {pet_path}. Error: {e}")
            print(f"Error processing sample at index {index}, path: {mri_path}. Error: {e}")

            # 可以选择重新抛出异常，或者返回一个特殊标记
            raise e
        

def get_dataset_from_pd(df: pd.DataFrame, transforms_fn: Transform, cache_dir: Optional[str]) -> Union[Dataset,PersistentDataset]: 
    """
    If `cache_dir` is defined, returns a `monai.data.PersistenDataset`. 
    Otherwise, returns a simple `monai.data.Dataset`.

    Args:
        df (pd.DataFrame): Dataframe describing each image in the longitudinal dataset.
        transforms_fn (Transform): Set of transformations
        cache_dir (Optional[str]): Cache directory (ensure enough storage is available)

    Returns:
        Dataset|PersistentDataset: The dataset

    Dataset

Dataset是一个基础的数据集类，主要用于将数据与相应的标签配对，并应用指定的转换操作。它直接从存储中读取数据，
每次访问数据时都会重新加载并应用转换操作。这种方式的优点是实现简单，但在处理大型数据集或复杂的转换操作时，
可能会导致数据加载和预处理的时间较长，从而影响训练速度。

PersistentDataset

PersistentDataset旨在通过缓存机制提高数据加载和预处理的效率。在首次运行时，它会将预处理后的数据存储在指定的缓存目录中。
在后续运行中，如果数据和转换操作未发生变化，PersistentDataset会直接从缓存中读取预处理后的数据，避免重复的预处理操作，
从而加快数据加载速度。这种方法特别适用于需要多次运行的实验，如超参数调优，或数据集较大、内存不足以一次性加载的情况。
    """
    assert cache_dir is None or os.path.exists(cache_dir), 'Invalid cache directory path'
    data = df.to_dict(orient='records')
    return Dataset(data=data, transform=transforms_fn) if cache_dir is None \
        else PersistentDataset(data=data, transform=transforms_fn, cache_dir=cache_dir)

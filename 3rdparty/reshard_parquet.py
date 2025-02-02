import os

import dask.dataframe as dd
from datasets import load_dataset

"""
YOU NEED `datasets` PACKAGE TO RUN THIS SCRIPT
"""

source = "data/fineweb/sample/10BT"
target = "data/reshard_fineweb_10bt"
desired_files = 1024

data = dd.read_parquet(os.path.join(source, "*.parquet"))
data = data.repartition(npartitions=desired_files)

os.makedirs(target, exist_ok=True)
data.to_parquet(os.path.join(target, ""), write_metadata_file=False)

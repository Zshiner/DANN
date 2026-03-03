from safetensors.torch import save_model
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
from config.config import Config
from scripts.deep import Comparator as DeepComparator
from scripts.mc import Comparator as MCComparator
from scripts.get_data import gd

# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#

llm_name = [
    'deepseek',
    'qwen',
    'zp'
]


if __name__ == '__main__':
    for (dataset_name, split_list) in gd.get_all():




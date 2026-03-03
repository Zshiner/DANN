import pandas as pd
import pathlib
import os

from sympy.unify.core import index


def run():
    base_dir = pathlib.Path(__file__).parent.parent / 'out'/ 'case'

    for root, dirs, files in os.walk(base_dir):
        break

    #
    for dataset in ['lung', 'sub_15_repeat_0']:
        all_data = None
        for file in files:
            if 'xlsx' in file and 'all' not in file and dataset in file:
                cur_data = pd.read_excel(base_dir / file, index_col=0)

                if all_data is None:
                    all_data = cur_data
                else:
                    model_name = pathlib.Path(file).stem.split('_')[-1]
                    all_data[model_name]=cur_data.iloc[:, -1]

        all_data.to_excel(base_dir / f'{dataset}_all.xlsx')

if __name__ == '__main__':
    run()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

resources_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


class Datasets:
    def __init__(self):
        files = ['train.csv', 'test.csv', 'sample_submission.csv']
        possible_roots = ['/kaggle/input/', './', resources_path]
        root_path = self.find_root_path(possible_roots, files)

        self.train_filepath = os.path.join(root_path, 'train.csv')
        self.test_filepath = os.path.join(root_path, 'test.csv')
        self.sample_filepath = os.path.join(root_path, 'sample_submission.csv')

        self.sample_df = pd.read_csv(self.sample_filepath)
        self.train_df = pd.read_csv(self.train_filepath)
        self.test_df = pd.read_csv(self.test_filepath)

    def get_sample_path(self) -> str:
        return self.sample_filepath

    def get_train_path(self) -> str:
        return self.train_filepath

    def get_test_path(self):
        return self.test_filepath

    def get_sample_df(self) -> pd.DataFrame:
        return self.sample_df

    def get_train_df(self) -> pd.DataFrame:
        return self.train_df

    def get_test_df(self) -> pd.DataFrame:
        return self.test_df

    @staticmethod
    def find_root_path(_possible_roots, _files):
        for _root_path in _possible_roots:
            for dirname, _, filenames in os.walk(_root_path):
                if set(_files).issubset(filenames):
                    if dirname:
                        root_dirname = os.path.join(_root_path, dirname)
                    else:
                        root_dirname = _root_path

                    return os.path.abspath(root_dirname)

        raise Exception(f"Could not find a good path, check your curdir {os.path.abspath(os.curdir)}")


class CPCDatasets:
    def __init__(self):
        cpc_df = pd.read_csv(os.path.join(resources_path, "cpc_202105.csv"))
        self.cpc_df = self._load_cpc_df(cpc_df)

    @staticmethod
    def _load_cpc_df(cpc_df) -> pd.DataFrame:
        cpc_df['class'] = cpc_df['sect_class'].apply(lambda x: x[1:])
        cpc_df['section_text'] = cpc_df['section_text'].apply(eval)
        cpc_df['class_text'] = cpc_df['class_text'].apply(eval)
        cpc_df['subclass_text'] = cpc_df['subclass_text'].apply(eval)
        cpc_df = cpc_df.rename({'sect_class': 'context'}, axis=1)
        cpc_df_grouped = cpc_df.groupby('context').agg(tuple)

        for col in cpc_df_grouped.columns:
            if 'subclass' not in col:
                cpc_df_grouped[col] = cpc_df_grouped[col].apply(lambda row: row[0])

        cpc_df = cpc_df_grouped.reset_index()
        cpc_df['subclass_text'] = cpc_df['subclass_text'].apply(lambda val: tuple([v for l in val for v in l]))
        return cpc_df

    def get_cpc_df(self) -> pd.DataFrame:
        return self.cpc_df

    def merge_with_df(self, _df) -> pd.DataFrame:
        _cpc_df = self.cpc_df.rename({'sect_class': 'context'}, axis=1)
        _cpc_merged_df = _df.merge(_cpc_df, on=['context'])
        _cpc_merged_df['section_cat'] = _cpc_merged_df['context'].apply(lambda x: ord(x[0]) - ord('A'))
        _cpc_merged_df['class_cat'] = _cpc_merged_df['context'].apply(lambda x: int(x[1:]))
        _cpc_merged_df['context_cat'] = _cpc_merged_df['context'].apply(lambda x: (ord(x[0]) - ord('A')) * 1000 + int(x[1:]))

        _cpc_merged_df['section_text'] = _cpc_merged_df.section_text.apply(lambda x: ",".join(x))
        _cpc_merged_df['class_text'] = _cpc_merged_df.class_text.apply(lambda x: ",".join(x))
        _cpc_merged_df['subclass_text'] = _cpc_merged_df.subclass_text.apply(lambda x: ",".join(x))
        _cpc_merged_df['title'] = _cpc_merged_df[['section_text', 'class_text']].apply(
            lambda row: f"{row['section_text']}; {row['class_text']}", axis=1
        )

        if 'score' in _cpc_merged_df.columns:
            _cpc_merged_df['score_map'] = _cpc_merged_df['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
        # More Transforms

        return _cpc_merged_df


if __name__ == '__main__':
    datasets = Datasets()
    print(datasets.get_sample_path())
    print(datasets.get_train_path())
    print(datasets.get_test_path())


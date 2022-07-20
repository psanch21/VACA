import os

from datasets._heterogeneous import HeterogeneousSCM
from utils.distributions import *

from aif360.datasets import GermanDataset
from sklearn.model_selection import KFold
import pandas as pd


class GermanSCM(HeterogeneousSCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples_tr: int = 800,
                 lambda_: float = 0.05,
                 transform=None,
                 ):
        assert split in ['train', 'valid', 'test']

        self.name = 'german'
        self.split = split
        self.num_samples_tr = num_samples_tr
        self.X_train = np.load(os.path.join(root_dir, 'german_data', 'train_1_X.npy'))
        self.X_valid = np.load(os.path.join(root_dir, 'german_data', 'test_valid_1_X.npy'))

        self.Y = None
        super().__init__(root_dir=root_dir,
                         transform=transform,
                         nodes_to_intervene=['age'],
                         structural_eq=None,
                         noises_distr=None,
                         nodes_list=['sex', 'age', 'R', 'S'],
                         adj_edges={'sex': ['R', 'S'],  # A#excluding 16
                                    'age': ['R', 'S'],  # Age
                                    'R': [],
                                    'S': [],
                                    },
                         lambda_=lambda_,
                         )

    @property
    def likelihoods(self):
        likelihoods_tmp = []
        # lik_node_sex = [self._get_lik('b',dim=1,normalize='dim')]
        # likelihoods_tmp.append(lik_node_sex)
        lik_node_sex = [self._get_lik('d', dim=1, normalize=None)]
        likelihoods_tmp.append(lik_node_sex)

        lik_node_age = [self._get_lik('d', dim=1, normalize='dim')]
        likelihoods_tmp.append(lik_node_age)

        lik_node_R = [self._get_lik('d', dim=2, normalize='dim')]
        likelihoods_tmp.append(lik_node_R)

        lik_node_S = [self._get_lik('c', dim=3, normalize='dim'),
                      self._get_lik('c', dim=5, normalize='dim'),
                      self._get_lik('c', dim=4, normalize='dim')]
        likelihoods_tmp.append(lik_node_S)

        return likelihoods_tmp

    def _create_data(self):
        X = np.concatenate([self.X_train, self.X_valid], axis=0)
        if self.split == 'train':
            self.X = X[:self.num_samples_tr]
        elif self.split == 'valid':
            num_samples = (X.shape[0] - self.num_samples_tr) // 2
            self.X = X[self.num_samples_tr:(self.num_samples_tr + num_samples)]
        elif self.split == 'test':
            num_samples = (X.shape[0] - self.num_samples_tr) // 2
            self.X = X[-num_samples:]
        self.Y = self.X[:, -1].copy()
        self.X = self.X[:, :-1]
        self.U = np.zeros([self.X.shape[0], 1])

    def node_is_image(self):
        return [False, False, False, False]

    def is_toy(self):
        return False

    def get_attributes_dict(self):
        # TODO:  Makes this more general
        unfair_attributes = list(range(2, 16))  # R, S
        fair_attributes = [1]  # Age
        sensitive_attributes = [0]  # Sex
        return {'unfair_attributes': unfair_attributes,
                'fair_attributes': fair_attributes,
                'sens_attributes': sensitive_attributes}

def prepare_german_datasets(data_dir):
    nodes_list = ['sex',  # A
                  'age',  # C
                  'credit_amount',  # R
                  'month',  # R repayment duration
                  'housing=A151', 'housing=A152', 'housing=A153',  # S
                  'savings=A61', 'savings=A62', 'savings=A63',
                  'savings=A64', 'savings=A65',  # S savings
                  'status=A11', 'status=A12',
                  'status=A13', 'status=A14']  ## S

    dataset = GermanDataset(protected_attribute_names=['sex'])
    dataset.labels = np.where(dataset.labels == 2, 0, 1)  # this is for y

    dataset.unfavorable_label = 0.0
    dataset.metadata['protected_attribute_maps']

    df = dataset.convert_to_dataframe()[0]

    X = df[nodes_list]
    y = df['credit']


    print("generating folds and saving them")
    inx = 1
    kf = KFold(n_splits=5)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("kf.split(X)", kf.split(X))
    for train_index, test_index in kf.split(X):
        X_train, X_test_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test_valid = y.iloc[train_index], y.iloc[test_index]
        data_train = pd.concat([X_train, y_train], axis=1).to_numpy()
        train_file = os.path.join(data_dir, f'train_{inx}_X.npy')
        np.save(train_file, data_train)  # save
        data_test_valid = pd.concat([X_test_valid, y_test_valid], axis=1).to_numpy()
        test_valid_file = os.path.join(data_dir, f'test_valid_{inx}_X.npy')
        np.save(test_valid_file, data_test_valid)  # save
        inx+=1


# def train_k_folds(kfold_idx):
#         print(f"Loading train_{kfold_idx}_X.npy")
#         data_train_fold = np.load(f'_data/german_kfold/train_{kfold_idx}_X.npy')
#         data_test_valid_fold = np.load(f'_data/german_kfold/test_valid_{kfold_idx}_X.npy')
#         X_train, y_train = data_train_fold[:, :-1], data_train_fold[:, -1]
#         X_test_valid, y_test_valid = data_test_valid_fold[:, :-1], data_test_valid_fold[:, -1]
#         X_train = pd.DataFrame(X_train)
#         y_train = pd.DataFrame(y_train)
#         X_test_valid = pd.DataFrame(X_test_valid)
#         y_test_valid = pd.DataFrame(y_test_valid)
#         return X_train, y_train, X_test_valid, y_test_valid




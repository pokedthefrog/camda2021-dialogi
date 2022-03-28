import pickle
import subprocess
import warnings
from pathlib import Path
from time import sleep

import networkx as nx
import numpy as np
from tqdm import tqdm, trange
from unidecode import unidecode


def get_project_root():
    return Path(__file__).absolute().parents[1]


def get_data_path(datafolder=''):
    proj_root = get_project_root()
    return proj_root/'dialogi'/'data'/datafolder


def stir_pickles(stem, suffix=0):   # Stem is str or tuple.
    picklejar_path = get_data_path('model_out')/'pickle_jar'
    stem = '-'.join(conv2tuple(stem))

    if suffix == 0:
        pkls = picklejar_path.glob(f'{stem}_*.pkl')

        suffixes = [pkl.stem.split('_')[1] for pkl in pkls]
        num_suffixes = [s for s in suffixes if s.isdigit()]
        if num_suffixes:
            # noinspection PyArgumentList
            suffix = np.array(num_suffixes, dtype=int).max()
            suffix += 1

    pkl_path = picklejar_path/f'{stem}_{suffix}.pkl'
    return pkl_path, pkl_path.is_file()


class PickleJar:
    @classmethod
    @property
    def cls_name(cls):    # noqa
        return str.lower(cls.__name__)

    @classmethod
    def build_stem(cls, stem_base=None, stem_sfx=None):
        pkl_stem = [stem_base if stem_base is not None else cls.cls_name]

        if stem_sfx is not None:
            stem_sfx = conv2tuple(stem_sfx)
            pkl_stem += stem_sfx

        return pkl_stem

    @classmethod
    def load(cls, stem_base=None, stem_sfx=None, sfx=0):
        pkl_stem = cls.build_stem(stem_base, stem_sfx)

        pkl_path, exists = stir_pickles(pkl_stem, sfx)
        if exists:
            with open(pkl_path, 'rb') as pkl:
                return pickle.load(pkl)
        warnings.warn("Loading failed: File could not be found.")

    def save(self, stem_base=None, stem_sfx=None, sfx=0):
        pkl_stem = type(self).build_stem(stem_base, stem_sfx)

        pkl_path, exists = stir_pickles(pkl_stem, sfx)
        if not exists:
            with open(pkl_path, 'wb') as pkl:
                pickle.dump(self, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            warnings.warn("Saving aborted: File already exists.")


class GenericSlot(PickleJar):
    def __init__(self, slot_data):
        self.data = slot_data


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def unidecode_pandas(df, cols):
    df_unidecoded = df.copy()

    df_unidecoded[cols] = df_unidecoded[cols].applymap(unidecode)
    return df_unidecoded


def sleep_progress(delay=600):
    for _ in trange(delay):
        sleep(1)

    print('')


def conv2tuple(str_input):
    if not isinstance(str_input, tuple):
        if isinstance(str_input, str):
            str_input = (str_input,)
        else:
            str_input = tuple(str_input)

    return str_input


def node2vec_snap_wrapper(nx_graph):
    path_node2vec = get_data_path('embedded')/'node2vec_snap'
    path_edges = path_node2vec/'temp'/'sim_net.edges'
    path_embed = path_node2vec/'temp'/'sim_net.embed'

    nx.write_weighted_edgelist(nx_graph, path=path_edges)
    p = subprocess.run(["./node2vec", "-i:temp/sim_net.edges", "-o:temp/sim_net.embed", "-w"], cwd=path_node2vec)
    p.check_returncode()

    emb_map = {}
    with open(path_embed) as f_emb:
        f_emb.readline()    # First line is header-- skip.
        for line in f_emb:
            node, vec = line.split(maxsplit=1)
            vec = np.fromstring(vec, dtype=float, sep=" ")

            emb_map[node] = vec

    # Clean up before returning.
    path_edges.unlink()
    path_embed.unlink()

    return emb_map


if __name__ == '__main__':
    pass

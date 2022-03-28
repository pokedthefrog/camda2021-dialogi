import copy
import gc
import warnings
from pathlib import Path

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras_tuner.engine import tuner_utils
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale, normalize

from dialogi import embeddings, utils
from dialogi.pubtator import PubTator
from dialogi.text_utils import UnivTextBlock


class CVHyperband(kt.BayesianOptimization):
    def _get_fold_checkpoint_path(self, trial_id, epoch, fold_idx):
        trial_epoch_path = Path(self._get_checkpoint_fname(trial_id, epoch))

        # Append (inner) fold index to path. KerasTuner saves the best model's weights per trial, which, in some cases,
        # are later retrieved to resume training. Here, multiple models are trained per trial. This avoids overwriting.
        fold_path = trial_epoch_path.parent/f'fold_{fold_idx}'/'checkpoint'
        return fold_path.resolve()

    def run_trial(self, trial, *args, **kwargs):
        # TODO: Uncomment this section if subclassing the Hyperband Tuner.
        # hp = trial.hyperparameters
        # if "tuner/epochs" in hp.values:
        #     kwargs.update({'epochs': hp.values['tuner/epochs'], 'initial_epoch': hp.values['tuner/initial_epoch']})
        callbacks = self._deepcopy_callbacks(kwargs.pop("callbacks", []))

        # Unpack the CV training & validation fold data.
        train_fold, valid_fold = args
        input_func, adapt_func = kwargs.pop('input_build_func'), kwargs.pop('model_adapt_func')

        # Loop over folds; build, fit, & validate model.
        histories = []       # Keras will take the mean.
        for fold_idx, train in train_fold.items():
            valid = valid_fold[fold_idx]

            # Building the input on demand saves memory.
            xy_train = input_func(train)
            xy_valid = input_func(valid)

            # Adapt text vectorisation on training data. This only has to be done on first epoch. Afterwards, the
            # Hyperband tuner creates checkpoints as needed and reloads layer weights when (re)building the model.
            # if kwargs['initial_epoch'] == 0:
            proc_texts = xy_train[0][0]
            adapt_func(proc_texts)

            # Get the fold-specific checkpoint callback.
            fold_cp_path = self._get_fold_checkpoint_path(trial.trial_id, self._reported_step, fold_idx)
            model_checkpoint = ModelCheckpoint(filepath=fold_cp_path, monitor=self.oracle.objective.name,
                                               mode=self.oracle.objective.direction, save_best_only=True,
                                               save_weights_only=True)

            for execution in range(self.executions_per_trial):
                copied_kwargs = copy.copy(kwargs)
                self._configure_tensorboard_dir(callbacks, trial, execution)

                callbacks.extend([tuner_utils.TunerCallback(self, trial), model_checkpoint])

                # Pass the training and validation data associated with each particular fold.
                copied_kwargs.update({'callbacks': callbacks, 'validation_data': xy_valid})
                fold_obj_vals = self._build_and_fit_model(trial, *xy_train, **copied_kwargs)

                histories.append(fold_obj_vals)

        # Garbage collection; this is very important, otherwise GPU runs out of memory after a couple of epochs.
        del train_fold, valid_fold, xy_train, xy_valid, input_func, adapt_func
        gc.collect()
        K.clear_session()

        return histories


class OneEpochTrain(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.stop_training = True


class Dialogi:
    def __init__(self, import_camda=True, data=None):
        self.data = {
            'raw':  None,   # Raw, imported data go here.
            'pub':  {},     # Keeps PubTator annotations.
            'utb':  {},     # Processed docs, as objects.
            'lbls': {},     # Specifies the class labels.
            'docs': {},     # Annotated docs, as strings.
            'vecs': {},     # Stores doc concept vectors.
        }

        if import_camda \
                and data is None:
            self._import_camda()
        else:
            # Meant to update 'raw' and 'lbls' data slots.
            self.data.update(data)

    def _import_camda(self):
        input_path = utils.get_data_path('camda_in')

        csv_kwargs = {'sep': '\t', 'header': 0, 'names': ['pmid', 'title', 'abstract'], 'keep_default_na': False}
        # noinspection PyTypedDict
        self.data['raw'] = {
            'pos':  pd.read_csv(input_path/'train-pos.tsv', **csv_kwargs),
            'neg':  pd.read_csv(input_path/'train-neg.tsv', **csv_kwargs),
            'val0': pd.read_csv(input_path/'valid-ds0.tsv', **csv_kwargs),
            'val1': pd.read_csv(input_path/'valid-ds1.tsv', **csv_kwargs),
        }
        # noinspection PyTypedDict
        self.data['lbls'] = {'class': {'pos': 1, 'neg': 0}, 'exclude': {}}

    def _get_data(self, slot='pub'):
        warn_map = {
            'raw':  "No imported data found.",
            'pub':  "No PubTator data found. To retrieve annotations, use 'call_pubtator'.",
            'utb':  "The imported documents need to be pre-processed. Call 'preproc_docs'.",
            'lbls': "No existing label mapping. Set data['lbls'] slot manually.",
            'docs': "Normalised, annotated text can be created by using 'norm_annot_docs'.",
            'vecs': "No feature vectors found. Use 'calc_concvecs' to calculate.",
        }

        data = self.data.get(slot)
        if not data:    # It can be None or an empty iterable.
            warnings.warn(warn_map[slot])

            return None
        return data

    def _save_slot(self, slot='pub', sfx=0):
        data = self._get_data(slot)
        if data is not None:

            for ds_name, slot_obj in data.items():
                slot_obj.save(slot, ds_name, sfx)

    def _load_slot(self, slot='pub', sfx='0'):
        class_map = {'pub': PubTator, 'utb': UnivTextBlock, }

        data_raw = self._get_data('raw')
        if data_raw is not None:

            for ds_name in data_raw:
                slot_cls = class_map.get(slot, utils.GenericSlot)
                slot_obj = slot_cls.load(slot, ds_name, str(sfx))
                if slot_obj:
                    self.data[slot][ds_name] = slot_obj

    def call_pubtator(self, **kwargs):
        data_raw = self._get_data('raw')
        if data_raw is not None:
            for ds_name, dataset in data_raw.items():
                pub_obj = self.data['pub'].get(ds_name)

                if pub_obj is None:
                    pub_obj = PubTator(dataset)
                    self.data['pub'][ds_name] = pub_obj

                pub_obj.request()
                pub_obj.retrieve_continuous(**kwargs)

    def save_pubtator(self, sfx=0):
        self._save_slot(slot='pub', sfx=sfx)

    def load_pubtator(self, sfx='0'):
        self._load_slot(slot='pub', sfx=sfx)

    def preproc_docs(self):
        data_pub = self._get_data('pub')
        if data_pub is not None:
            for ds_name, pub_obj in data_pub.items():
                utb_obj = self.data['utb'].get(ds_name)

                if utb_obj is None:
                    # List of merged Titles & Abstracts.
                    doc_texts = pub_obj.export_docs()[0]

                    utb_obj = UnivTextBlock(doc_texts)
                    self.data['utb'][ds_name] = utb_obj

    def save_procdocs(self, sfx=0):
        self._save_slot(slot='utb', sfx=sfx)

    def load_procdocs(self, sfx='0'):
        self._load_slot(slot='utb', sfx=sfx)

    def norm_annot_docs(self, coarse_tagging=False, **kwargs):
        data_utb = self._get_data('utb')
        data_pub = self._get_data('pub')

        if data_utb is not None \
                and data_pub is not None:

            for ds_name, utb_obj in data_utb.items():
                pub_obj = data_pub.get(ds_name)

                if pub_obj is None:
                    warnings.warn("Mismatched data.")
                    self.data['docs'] = {}     # Revert any changes.
                    return

                doc_annots = pub_obj.export_docs(coarse_tagging)[1]

                docs = utb_obj.norm_annot_docs(doc_annots, **kwargs)
                self.data['docs'][ds_name] = utils.GenericSlot(docs)

    def save_proctxts(self, sfx=0):
        self._save_slot(slot='docs', sfx=sfx)

    def load_proctxts(self, sfx='0'):
        self._load_slot(slot='docs', sfx=sfx)

    def calc_concvecs(
            self, source_concepts=('MESH_Chemical', 'MESH_Disease', 'CC_Chemical', 'CTDBase_Disease',), coarse=True
    ):
        embeddings_map = {
            'MeanVecs': 'embeddings.MeanVectors(concept, mesh)',
            'CC':       'embeddings.ChemicalChecker(mesh)',
            'StringDB': 'embeddings.ChemStringGraph(mesh)',
            'MESH':     'embeddings.ChemDisMeshTree(concept)',
            'CTDBase':  'embeddings.DisCTDBaseGraph(mesh)',
        }

        data_pub = self._get_data('pub')
        if data_pub is not None:
            sources = {each.split('_')[0] for each in source_concepts}

            ds_dict = {ds_name: {source: {} for source in sources} for ds_name in data_pub}
            embedding_source_spaces = {source: {} for source in sources}

            for source_concept in source_concepts:
                source, concept = source_concept.split('_')

                # First inner loop collects every seen MeSH ID.
                mesh = set()
                for _, pub_obj in data_pub.items():
                    mesh.update(pub_obj.annots2docs[concept])
                mesh = tuple([m.split(':')[1] for m in mesh])     # Formatted as: 'MESH:#######'; extract the IDs.

                embeddings_obj = eval(embeddings_map[source])     # Initialise concept-specific embeddings object.
                embeddings_obj.populate()

                if not coarse:
                    embedding_source_spaces[source].update(embeddings_obj.mesh2vec_dict)
                    continue         # Do not calculate average embeddings for documents.

                # Second inner loop iterates over all datasets.
                for ds_name, pub_obj in data_pub.items():
                    annots_w_freqs = pub_obj.docs2annots                   # Document annotations w/ frequencies.
                    vecs = embeddings_obj.get_mean_vecs(annots_w_freqs)    # Document frequency-weighted vectors.

                    ds_dict[ds_name][source].update(vecs)

            # If coarse embeddings has been set to False, return now, as there are no average embeddings to save.
            if not coarse:
                return embedding_source_spaces

            # Wrap as 'GenericSlot' to enable saving & loading.
            for ds_name, ds_data in ds_dict.items():
                self.data['vecs'][ds_name] = utils.GenericSlot(ds_data)

    def concat_spaces(self, src_space_array, name='space_concat', normalise=True):
        data_vec = self._get_data('vecs')
        if data_vec is not None:
            for ds_name in data_vec:
                if not self.data['vecs'][ds_name].data.get(name):
                    self.data['vecs'][ds_name].data[name] = {}

                vec_spaces = []
                for src_space in src_space_array:
                    source, space = src_space.split('_')

                    vecs = list(data_vec[ds_name].data[source][space].values())
                    if normalise:
                        vecs = normalize(vecs)

                    vec_spaces.append(vecs)
                concat_vecspace = np.hstack(vec_spaces)

                ds_dict = {}
                for row, array in enumerate(concat_vecspace):
                    ds_dict[row] = array
                self.data['vecs'][ds_name].data[name]['concat'] = ds_dict

    def save_concvecs(self, sfx=0):
        self._save_slot(slot='vecs', sfx=sfx)

    def load_concvecs(self, sfx='0'):
        self._load_slot(slot='vecs', sfx=sfx)

    def create_cvfolds(self, num_outer_folds=5, num_inner_folds=3, seed=37):
        data_lbls = self._get_data('lbls')
        data_docs = self._get_data('docs')

        if None in (data_lbls, data_docs):
            return None

        if data_lbls.get('class') is None:
            warnings.warn("No class datasets are specified-- do this by setting the data['lbls'] slot manually.")
            return None

        agg_keys, agg_lbls = [], []
        label2dsname_map = {}
        for ds_name, ds_lbl in data_lbls['class'].items():
            if ds_name not in data_docs:
                warnings.warn("Mismatched classes.")
                return None

            ds_keys = data_docs[ds_name].data.keys()
            ds_lbls = [ds_lbl, ] * len(ds_keys)
            agg_keys.extend(ds_keys)
            agg_lbls.extend(ds_lbls)

            label2dsname_map[ds_lbl] = ds_name

        agg_keys = np.array(agg_keys)
        agg_lbls = np.array(agg_lbls)

        outer_skf = StratifiedKFold(n_splits=num_outer_folds, shuffle=True, random_state=seed)
        inner_skf = StratifiedKFold(n_splits=num_inner_folds, shuffle=True, random_state=seed)

        nested_folds = {}

        out_kfolds = outer_skf.split(agg_keys, agg_lbls)
        for out_k, (out_train_idx, test_idx) in enumerate(out_kfolds):
            out_train_keys, test_keys = agg_keys[out_train_idx], agg_keys[test_idx]
            out_train_lbls, test_lbls = agg_lbls[out_train_idx], agg_lbls[test_idx]

            nested_folds[out_k] = {'test':  {'keys': test_keys, 'lbls': test_lbls}, 'train': {}, 'valid': {}, }

            in_kfolds = inner_skf.split(out_train_keys, out_train_lbls)
            for in_k, (train_idx, valid_idx) in enumerate(in_kfolds):
                train_keys, valid_keys = out_train_keys[train_idx], out_train_keys[valid_idx]
                train_lbls, valid_lbls = out_train_lbls[train_idx], out_train_lbls[valid_idx]

                in_kfold_train = {in_k: {'keys': train_keys, 'lbls': train_lbls, }}
                in_kfold_valid = {in_k: {'keys': valid_keys, 'lbls': valid_lbls, }}

                # noinspection PyTypeChecker
                nested_folds[out_k]['train'].update(in_kfold_train)
                # noinspection PyTypeChecker
                nested_folds[out_k]['valid'].update(in_kfold_valid)

        train_test_split = {'label2dsname_map': label2dsname_map, 'nested_folds': nested_folds, }
        return train_test_split

    def build_fold_input(
            self, fold_data, scale_mean=False, scale_std=False, svd=False, svd_dim=128, normalise=True, svd_seed=37
    ):
        data_lbls = self._get_data('lbls')
        data_docs = self._get_data('docs')
        data_vecs = self._get_data('vecs')

        if data_lbls is not None \
                and data_vecs is not None:

            if isinstance(fold_data, str):  # This is here for building external validation datasets.
                label2dsname_map = {0: fold_data, }    # Here, fold_data holds the ext. dataset name.

                keys = list(self.data['docs'][fold_data].data)
                ext_labels = [0]*len(keys)
                fold_data = {'keys': keys, 'lbls': ext_labels}
            else:
                label2dsname_map = {label: ds_name for ds_name, label in data_lbls['class'].items()}

            # The input array is saved temporarily as dictionary. It contains texts and 3 different vector sources
            # (CTD, CC, and MeSH) as keys. Within each source, alternative vector spaces occupy an extra dimension.
            # TODO: (A) Skipping unused inputs-- proper implementation needed.
            to_skip = ['CTDBase', 'CC', 'MESH']
            input_keys = ['Text'] + list(data_vecs[next(iter(data_vecs))].data)
            model_input_array = {k: [] for k in input_keys if k not in to_skip}

            for key_idx, ds_label in enumerate(fold_data['lbls']):
                ds_name = label2dsname_map[ds_label]
                doc_idx = fold_data['keys'][key_idx]

                doc = data_docs[ds_name].data.get(doc_idx)
                if doc is None:
                    warnings.warn("Mismatched data.")
                    return None
                model_input_array['Text'].append(doc)

                for source_name, source_dict in data_vecs[ds_name].data.items():
                    # TODO: (B) Again, skip unused inputs.
                    if source_name in to_skip:
                        continue

                    space_vecs = []
                    for space_dict in source_dict.values():
                        vector = space_dict[doc_idx]
                        space_vecs.append(vector)
                    model_input_array[source_name].append(np.vstack(space_vecs))

            model_input_array = {k: np.stack(v, axis=0) for k, v in model_input_array.items()}
            model_input_array = list(model_input_array.values())

            # Perform dimensionality reduction using truncated SVD, if requested. Optionally, scale and normalise.
            # Only apply that to concatenated spaces-- that is, exclude the first four elements in the input list.
            # TODO: (C) Because we've dropped inputs, only the first element (texts) is skipped.
            for array_idx, in_array in enumerate(model_input_array[1:]):
                out_array = np.array(in_array)
                if svd:
                    out_array = np.zeros([*in_array.shape[0:2], svd_dim])

                # Iterate over different spaces.
                for space_idx in range(in_array.shape[1]):
                    space_array = in_array[:, space_idx, :]

                    if scale_mean or scale_std:
                        space_array = normalize(space_array)
                        space_array = scale(space_array, with_mean=scale_mean, with_std=scale_std)
                    if svd:
                        trunc_svd = TruncatedSVD(n_components=svd_dim, random_state=svd_seed)
                        space_array = trunc_svd.fit_transform(space_array)
                    if normalise:
                        space_array = normalize(space_array)

                    out_array[:, space_idx, :] = space_array
                model_input_array[array_idx+1] = out_array

            return model_input_array, fold_data['lbls']
        return None

    def train_test_model(self, cv_folds, model_obj, max_trials=60, batch_size=32, seed=37, project_name='lstm_nf_ef'):
        helper_funcs = {
            'input_build': self.build_fold_input,           # Turns fold-specific indices into actual data.
            'model_build': model_obj.get_model,             # Compiles the (baseline) model for KerasTuner.
            'model_adapt': model_obj.adapt_vector_layer,    # Adapts the text vectorisation layer.
        }

        ncv_results = {
            'winner_hp': [],
            'nf': {'winners': [], 'metrics': [], 'f1@threshold': {'x': [], 'y': []}, },
            'ef': {'winners': [], 'metrics': [], 'f1@threshold': {'x': [], 'y': []}, },
        }

        for outer_fold_idx, fold_data in cv_folds['nested_folds'].items():
            # For each outer fold, we are restarting the process-- to avoid overwriting, append fold index.
            proj_name = f'{project_name}-fold_{outer_fold_idx}'
            basepath = utils.get_data_path('model_out') / 'predictions_ext'

            # Initialise the CV-Tuner.
            cv_tuner = CVHyperband(hypermodel=helper_funcs['model_build'], objective='val_loss', max_trials=max_trials,
                                   seed=seed, directory='kt_ncvbayes', project_name=proj_name, num_initial_points=None)

            # Select best inner model.
            cv_tuner.search(fold_data['train'], fold_data['valid'], batch_size=batch_size, callbacks=[OneEpochTrain()],
                            model_adapt_func=helper_funcs['model_adapt'], input_build_func=helper_funcs['input_build'])

            # Build the training set by merging inner folds.
            train_keys_concat = np.concatenate([fold_data['train'][0]['keys'], fold_data['valid'][0]['keys']])
            train_lbls_concat = np.concatenate([fold_data['train'][0]['lbls'], fold_data['valid'][0]['lbls']])
            xy_train = self.build_fold_input({'keys': train_keys_concat, 'lbls': train_lbls_concat})
            xy_test = self.build_fold_input(fold_data['test'])

            # Build the model with the best hyperparameters.
            proc_texts = xy_train[0][0]
            model_obj.adapt_vector_layer(proc_texts)

            winner_hp = cv_tuner.get_best_hyperparameters(num_trials=1)[0]
            nf_winner = cv_tuner.hypermodel.build(winner_hp)

            # Fit on inner data and evaluate on the test set.
            nf_history = nf_winner.fit(*xy_train, validation_data=xy_test, batch_size=batch_size, epochs=1)

            # Gather inner winner's layer shapes and weights.
            base_weights, base_shapes = [], []
            for layer_idx, layer in enumerate(nf_winner.layers):
                base_weights.append(layer.get_weights())

                # For shapes, only interested in layers #2-4.
                if layer_idx < 2 \
                        or layer_idx > 4:
                    continue
                layer_shape = layer.get_output_at(0).get_shape().as_list()
                base_shapes.append([dim for dim in layer_shape if dim][0])
            model_obj.set_shape(base_shapes)

            # Transfer weights and compile the extended model.
            ef_winner = model_obj.get_model_ef()
            extended_layers = [layer for layer in ef_winner.layers]
            extended_layers[1].set_weights(base_weights[1])    # Text vectorisation
            extended_layers[2].set_weights(base_weights[2])    # Text embeddings
            extended_layers[4].set_weights(base_weights[3])    # Bidirectional-LSTM
            extended_layers[6].set_weights(base_weights[4])    # Dense #1 (ReLU)
            extended_layers[9].set_weights(base_weights[6])    # Dense #2 (Sigmoid)

            for layer_idx in [1, 2, 4, 6, 9]:
                extended_layers[layer_idx].trainable = False

            ef_winner = model_obj.compile_model_default(ef_winner)
            ef_history = ef_winner.fit(*xy_train, validation_data=xy_test, batch_size=batch_size, epochs=1)

            # Temporary hold models, alongside fitting & testing histories.
            models_tmp = {
                'nf': {'model': nf_winner, 'history': nf_history.history, 'threshold': 0.5},
                'ef': {'model': ef_winner, 'history': ef_history.history, 'threshold': 0.7},
            }

            # Save results for internal & external validation.
            ncv_results['winner_hp'].append(winner_hp)
            for model_type in ['nf', 'ef']:
                model, history = models_tmp[model_type]['model'], models_tmp[model_type]['history']
                ncv_results[model_type]['winners'].append(model)
                ncv_results[model_type]['metrics'].append({met: vals[0] for met, vals in history.items()})

                for met_name, met_value in ncv_results[model_type]['metrics'][-1].items():
                    if 'f1_score_' not in met_name:
                        continue

                    threshold = round(float(met_name.split('_')[-1]), 6)
                    ncv_results[model_type]['f1@threshold']['x'].append(threshold)
                    ncv_results[model_type]['f1@threshold']['y'].append(met_value)

                # Predict on the external validation datasets.
                for extval_set in ['val0', 'val1']:
                    x_extval = self.build_fold_input(extval_set)[0]

                    pred_probs = model.predict(x_extval, batch_size=batch_size)
                    pred_bool = np.where(pred_probs > models_tmp[model_type]['threshold'], 1, 0).flatten()

                    save_path = basepath / f'{proj_name}--{extval_set}-{model_type}.txt'
                    with open(save_path, 'w') as f_preds:
                        for pred in pred_bool:
                            f_preds.write(str(pred)+'\n')

        return ncv_results


if __name__ == '__main__':
    pass

import csv
import re
import warnings
from collections.abc import Iterable
from itertools import groupby

import numpy as np
import stanza
from flatten_dict import flatten, unflatten

from dialogi import utils

stopword_path = utils.get_data_path('embedded')/'stop_words.csv'
with open(stopword_path) as fl:
    reader = csv.reader(fl)
    STOPWORDS = next(reader)
UPOS_SKIP = 'ADP,AUX,CCONJ,DET,INTJ,NUM,PART,PRON,PROPN,PUNCT,SCONJ,SYM,X'.split(',')


class UnivTextBlock(utils.PickleJar):
    _stanza = stanza.Pipeline(package='genia', processors='tokenize,lemma,pos,depparse', verbose=False, use_gpu=False)

    def __init__(self, docs_raw=None):
        self._docs = None

        if docs_raw is not None:
            self.import_docs(docs_raw)

    def import_docs(self, docs_raw):
        docs = self._conv2stanza_docs(docs_raw)
        docs_preproc = type(self)._stanza(docs)
        self._process(docs_preproc)

    @staticmethod
    def _conv2stanza_docs(docs):
        if not isinstance(docs, Iterable):
            docs = (docs,)

        docs_stanza = [stanza.Document([], text=text) for text in docs]
        return docs_stanza

    def _process(self, docs_preproc, use_abs_wpos=True):    # use_abs_wpos=False will break things.
        docs_tmp = {'docs': {}, 'tags': {}, }
        for doc_idx, doc in enumerate(docs_preproc):

            doc_sens_tmp = {'sens': {}, 'tags': {}, }
            for sen_idx, sen in enumerate(doc.sentences):
                sen_tmp = {
                    'text':  sen.text,
                    'tags':  {},
                    'words': {},
                    'w_pos': {},                # Word position.
                    'w_map': {0: 'root', },     # Map stanza to internal ids.
                    'w_rel': [],                # Grammatical relations.
                }

                wpos_ref = 0
                for word_idx, word in enumerate(sen.words):
                    word_tmp = {key: getattr(word, key) for key in ['id', 'text', 'lemma', 'pos']}

                    w_pos = np.array([word.start_char, word.end_char])
                    if (not use_abs_wpos) \
                            and word_idx == 0:
                        wpos_ref = w_pos[0]

                    word_tmp.update({'w_pos': w_pos - wpos_ref, 'tags': set(), })

                    sen_tmp_update = {
                        'words': {word_idx:       word_tmp},
                        'w_map': {word_tmp['id']: word_idx},
                        'w_pos': {
                            word_tmp['w_pos'][0]: {
                                'end':  word_tmp['w_pos'][1],
                                'len':  np.diff(word_tmp['w_pos'])[0],
                                'idx':  word_idx,
                                'word': word_tmp,
                            }
                        }, }
                    sen_tmp = self._update_multidim_dict(sen_tmp, sen_tmp_update)

                for dep in sen.dependencies:
                    w_rel = ((sen_tmp['w_map'][dep[0].id], sen_tmp['w_map'][dep[2].id]), dep[1],)
                    sen_tmp['w_rel'].append(w_rel)

                doc_sens_tmp['sens'][sen_idx] = sen_tmp
            docs_tmp['docs'][doc_idx] = doc_sens_tmp

        self._docs = docs_tmp

    @staticmethod
    def _update_multidim_dict(target_dict, update_dict):
        target_dict = flatten(target_dict, keep_empty_types=(list, dict,))
        update_dict = flatten(update_dict)

        target_dict.update(update_dict)

        updated_dict = unflatten(target_dict)
        return updated_dict

    def _get_doc(self, doc_idx):
        doc = self._docs['docs'].get(doc_idx)
        return doc

    def _get_sen(self, doc_idx, sen_idx):
        doc = self._get_doc(doc_idx)

        sen = None
        if doc:
            sen = doc['sens'].get(sen_idx)

        return sen

    def _get_sen_wpos_bounds(self, *args):
        sen = self._get_sen(*args)
        wpos_lb = tuple(sen['w_pos'])[0]
        wpos_rb = tuple(sen['w_pos'].values())[-1]['end']

        wpos_limits = np.array([wpos_lb, wpos_rb])
        return wpos_limits

    def _get_word(self, w_idx, **kwargs):
        sen = self._get_sen(**kwargs)

        word = None
        if sen:
            word = sen['words'].get(w_idx)

        return word

    def tag_word(self, doc_idx, sen_idx, w_pos, w_tag='*-tagged', sen_tag=None, threshold=0.9):
        if sen_tag is None:
            sen_tag = w_tag

        sen_tmp = self._get_sen(doc_idx, sen_idx)
        if sen_tmp is None:
            warnings.warn("Incorrect document or sentence ID, tagging failed.")
            return

        widx_tagged = []
        widx_best = None
        overlap_max = 0
        for wpos_start, wpos_dict in sen_tmp['w_pos'].items():
            wpos_cand = (wpos_start, wpos_dict['end'])
            overlap = self._calc_range_overlap(w_pos, wpos_cand)/wpos_dict['len']
            if overlap == 0:
                continue

            # Update the best candidate word.
            if overlap_max < overlap:
                widx_best = wpos_dict['idx']
                overlap_max = overlap

            widx = (wpos_dict['idx'], True if overlap >= threshold else False,)
            widx_tagged.append(widx)
        # Keep words above threshold OR single best hit (can be below threshold).
        widx_tagged = [w[0] for w in widx_tagged if (w[1] or w[0] == widx_best)]

        if overlap_max < threshold:
            if not widx_tagged:
                warnings.warn("No overlapping words were found, tagging failed.")
                return
            warnings.warn(f"Threshold wasn't met; best hit tagged with overlap {overlap_max}.")

        # Tag on batch, per-document, and sentence levels.
        tags_tmp_update = (self._docs, self._get_doc(doc_idx), sen_tmp,)
        for tag_idx, tag_tmp in enumerate(tags_tmp_update):
            if sen_tag not in tag_tmp['tags']:
                tag_tmp['tags'][sen_tag] = set()

            if tag_idx == 2:
                tag_tmp['tags'][sen_tag].update(widx_tagged)
                continue
            tag_tmp['tags'][sen_tag].add(sen_idx)

        # Append tags on per-word level.
        for w_idx in widx_tagged:
            word_tmp = self._get_word(w_idx, doc_idx=doc_idx, sen_idx=sen_idx)
            word_tmp['tags'].add(w_tag)

    @staticmethod
    def _calc_range_overlap(range1, range2):
        overlap = max(0, min(range1[1], range2[1]) - max(range1[0], range2[0]))
        return overlap

    def _calc_annot_overlap(self, doc_idx, sen_idx, annot_dict):
        annot_bounds = np.array(annot_dict['w_pos'])
        annot_len = np.diff(annot_bounds)[0]

        sen_bounds = self._get_sen_wpos_bounds(doc_idx, sen_idx)
        overlap = self._calc_range_overlap(sen_bounds, annot_bounds)/annot_len

        return overlap

    def norm_annot_docs(self, annot_data, upos_skip=None, stopwords=None, treat_tags='rep'):
        if upos_skip is None:
            upos_skip = UPOS_SKIP
        if stopwords is None:
            stopwords = STOPWORDS

        # Tag.
        for doc_idx, annots in annot_data.items():
            sidx_iter = iter(self._get_doc(doc_idx)['sens'])

            sen_idx = next(sidx_iter)
            for _, annot_dict in enumerate(annots):
                # Iterate through sentences until we find the one with the annotated word.
                while True:
                    overlap = self._calc_annot_overlap(doc_idx, sen_idx, annot_dict)

                    if overlap == 0:
                        sen_idx = next(sidx_iter)
                        continue

                    if overlap == 1:
                        self.tag_word(doc_idx, sen_idx, **annot_dict)
                        break

                    # Reaching here means that: 0< overlap <1. This tends to happen when sentence segmentation fails,
                    # thus incorrectly splitting a sentence in two, and the annotated term just happens to fall where
                    # this error has occurred.
                    max_sidx = next(reversed(self._get_doc(doc_idx)['sens']))
                    in_sen_idx = sen_idx
                    while True:
                        in_sen_idx += 1
                        if in_sen_idx > max_sidx:
                            break

                        overlap = self._calc_annot_overlap(doc_idx, in_sen_idx, annot_dict)
                        if overlap == 0:
                            break

                        self.tag_word(doc_idx, in_sen_idx, **annot_dict)
                    break

        # Annotate and normalise.
        docs_strproc_tmp = {}
        for doc_idx in annot_data:
            docs_strproc_tmp[doc_idx] = []

            sens = self._get_doc(doc_idx)['sens']
            for _, sen_dict in sens.items():

                for _, word_dict in sen_dict['words'].items():
                    # Remove special characters at the beginning or end.
                    lemma = re.sub(r'^\W+|\W+$', '', word_dict['lemma'])
                    upos = word_dict['pos']

                    if lemma in stopwords or upos in upos_skip \
                            or len(lemma) < 3:
                        continue

                    tags_norm = sorted(map(str.lower, word_dict['tags']))
                    tag = ' '.join(tags_norm)

                    if tag:
                        if treat_tags == 'del':
                            continue
                        if treat_tags == 'rep':
                            docs_strproc_tmp[doc_idx].append(tag)
                            continue
                    # If there are no tags, or they should be skipped.
                    docs_strproc_tmp[doc_idx].append(lemma)

        # Remove consecutive lemmas, then rebuild to string, & return.
        prune_consec = lambda w_array: ' '.join(w for w, _ in groupby(w_array))
        docs_strproc = {k: prune_consec(d) for k, d in docs_strproc_tmp.items()}

        return docs_strproc


if __name__ == '__main__':
    pass

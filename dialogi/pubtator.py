import re

import numpy as np
import requests

from dialogi import utils

PUBTATOR_BASE_URL = 'https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/'
PUBTATOR_RE = {
    'title': re.compile(r'^([0-9]+)\|t\|(.*)$'),
    'abstr': re.compile(r'^([0-9]+)\|a\|(.*)$'),
    'annot': re.compile(r'^([0-9]+)\t([0-9]+)\t([0-9]+)\t([^\t]+)\t([^\t]+)\t*([^\t]*)$'),
}


class PubTator(utils.PickleJar):
    def __init__(self, doc_df):
        # Expects columns: 'pmid', 'title', and 'abstract'.
        self._docs = doc_df.copy()

        # Clean up text by decoding unicode chars to ASCII.
        self._docs = utils.unidecode_pandas(self._docs, ['title', 'abstract'])

        # ID is a unique per-document PMID-like identifier.
        index_str = self._docs.index.astype(str)
        self._docs[['id', 'annots']] = [index_str.str.zfill(8), None]

        self._annot2doc_map = {}
        self._session_ids = set()

    def request(self, bio_concepts='All', chunk_size=20, cleanup_ids=True):
        # Normally, requests should only be called once. When called twice, assume something has gone wrong and purge
        # all active sessions. Needed in the rare occurrence of a session getting stuck, thus never returning results.
        if cleanup_ids:
            self._session_ids.clear()

        docs = self._docs.loc[self._docs['annots'].isna()]

        group_indices = np.arange(len(docs)) // chunk_size
        group_num = np.unique(group_indices).size
        print(f"Submitting documents split in {group_num} groups:")

        for group_idx, grouped_docs in docs.groupby(group_indices):
            doc_str = ''
            for doc_idx, doc in grouped_docs.iterrows():
                doc_str += f"{doc.id}|t|{doc.title}\n{doc.id}|a|{doc.abstract}\n\n"

            req_url = '/'.join([PUBTATOR_BASE_URL, 'submit', bio_concepts])

            resp = requests.post(req_url, data=doc_str)
            resp.raise_for_status()      # Raise error for non-200 response.

            session_id = resp.text
            self._session_ids.add(session_id)

            print(f"\tGroup {group_idx+1}/{group_num} has been submitted.", end='\r')
        print("\n\t\tRequest completed.\n")

    def retrieve(self):
        act_session_num = len(self._session_ids)
        print(f"Attempting to retrieve {act_session_num} document groups:")

        for session_idx, session_id in enumerate(tuple(self._session_ids)):
            print(f"\tProcessing group {session_idx+1}/{act_session_num}.", end='\r')

            req_url = '/'.join([PUBTATOR_BASE_URL, 'retrieve', session_id])
            resp = requests.get(req_url)

            if resp.status_code == 200:
                self._session_ids.remove(session_id)
                annots_raw = resp.text

                cur_doc_id = None
                for line in annots_raw.splitlines():
                    if line:
                        m = PUBTATOR_RE['title'].search(line)
                        if m:
                            doc_id = m.group(1)
                            assert not self._append_doc_annot(doc_id, None)

                            cur_doc_id = doc_id
                            continue

                        m = PUBTATOR_RE['abstr'].search(line)
                        if m:
                            doc_id = m.group(1)
                            assert cur_doc_id == doc_id
                            continue

                        m = PUBTATOR_RE['annot'].search(line)
                        if m:
                            doc_id = m.group(1)
                            assert cur_doc_id == doc_id

                            groups = m.groups()
                            w_pos = np.array(groups[1:3], dtype='int32')
                            text, concept_type, concept_id = groups[3:6]

                            if not concept_id \
                                    or concept_id == '-':
                                continue

                            annot = {'w_pos': w_pos, 'w_tag': concept_id, 'sen_tag': concept_type}
                            self._append_doc_annot(cur_doc_id, annot)

                            self._map_annot_to_doc(concept_type, concept_id, cur_doc_id)

        rem_session_num = len(self._session_ids)
        ret_session_num = act_session_num - rem_session_num
        print(f"""
        \r\t\tRetrieval completed;
        \r\t\t\t\u2022 Retrieved: {ret_session_num} groups,
        \r\t\t\t\u2022 Remaining: {rem_session_num} groups.
        """)

    def _append_doc_annot(self, doc_id, annot=None):
        doc = self._docs[self._docs.id == doc_id]
        idx = doc.index[0]

        # Passing None initialises, if needed, and returns past status.
        if annot is None:
            was_init = True

            if doc.annots.iat[0] is None:
                self._docs.at[idx, 'annots'] = []
                was_init = False

            return was_init

        self._docs.at[idx, 'annots'].append(annot)

    def _map_annot_to_doc(self, annot_type, annot_id, doc_id):
        if not self._annot2doc_map.get(annot_type):
            self._annot2doc_map[annot_type] = {}

        if not self._annot2doc_map[annot_type].get(annot_id):
            self._annot2doc_map[annot_type][annot_id] = set()

        self._annot2doc_map[annot_type][annot_id].add(doc_id)

    def retrieve_continuous(self, total_wait=1200, wait_interval=60):
        num_attempts = total_wait // wait_interval
        for attempt_idx in range(0, num_attempts):
            if attempt_idx != 0:
                print("Sleeping before retrying:")
                utils.sleep_progress(wait_interval)

            self.retrieve()

            if not self._session_ids \
                    or not self.remaining_docs:
                print("\t\tAll groups retrieved!")
                # Clean up remaining sessions. Sometimes, the server refuses to return results for a (seemingly) active
                # session. When this happens, a new request has to be made. Cleaning up here should only be needed when
                # calling 'request' with the option cleanup_ids=False.
                self._session_ids = set()
                return
        print("\t\tMaximum number of attempts was reached, but not all groups were retrieved.")

    @property
    def annotated_docs(self):
        annot_slice = self._docs.id[~self._docs.annots.isna()]
        return annot_slice.to_dict()

    @property
    def remaining_docs(self):
        remain_slice = self._docs.id[self._docs.annots.isna()]
        return remain_slice.to_dict()

    def export_docs(self, coarse_tagging=False):
        # Sets w_tag equal to sen_tag. Annotate words with concept types instead of MeSH IDs.
        def conv2coarse(annot_dict):
            annot = dict(annot_dict)
            annot['w_tag'] = annot['sen_tag']
            return annot

        merged = self._docs.loc[:, ['title', 'abstract']].apply(lambda x: ' '.join(x), axis=1)

        annots = self._docs['annots']
        if coarse_tagging:
            annots = annots.apply(lambda annot: list(map(conv2coarse, annot)))

        return merged.to_list(), annots.to_dict()

    @property
    def docs_raw_df(self):
        return self._docs

    @property
    def annots2docs(self):
        return self._annot2doc_map

    @property
    def concepts2ids(self):
        concept_types = tuple(self._annot2doc_map)

        concept2id_map = {}
        for each in concept_types:
            ids = tuple(self._annot2doc_map[each])
            concept2id_map[each] = ids

        return concept2id_map

    @property
    def docs2annots(self):
        annots = self._docs['annots']
        concepts = tuple(self._annot2doc_map)

        # Pre-allocate dict structure.
        doc2annotfreq_map = {idx: {c: [] for c in concepts} for idx in annots.index}
        for doc_idx, doc_annots in annots.iteritems():
            # Append all seen terms.
            for annot in doc_annots:
                concept_type, concept_id = annot['sen_tag'], annot['w_tag']
                doc2annotfreq_map[doc_idx][concept_type].append(concept_id)

            # Calculate frequencies.
            for concept, concept_ids in doc2annotfreq_map[doc_idx].items():
                unq_ids, freqs = np.unique(np.array(concept_ids), return_counts=True)
                if freqs.size != 0:
                    freqs = freqs / np.array(concept_ids).size
                doc2annotfreq_map[doc_idx][concept] = dict(zip(unq_ids, freqs))

        return doc2annotfreq_map


if __name__ == '__main__':
    pass

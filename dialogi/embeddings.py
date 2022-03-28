import pickle
import warnings

import faiss
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import requests
import xmltodict
from node2vec import Node2Vec
from scipy.spatial import distance
from signaturizer import Signaturizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from dialogi import net_utils
from dialogi import utils


def get_mesh2structure_map():
    chem_mesh = ChemDisMeshTree(concept='Chemical')

    desc, supp = chem_mesh.get_trees()
    mesh_map = desc | supp

    # MeSH term(s) --[MeSH]--> MeSH ID (many-to-1)
    term2mesh_map = {}
    for mesh, data in mesh_map.items():
        concept_terms = data['concepts']['terms']
        for terms in concept_terms:
            for term in terms:
                term2mesh_map[term.lower()] = mesh

    # MeSH term --[ChemIDplus]--> CIP ID
    term2cipid_map = {}
    base_url = 'https://chem.nlm.nih.gov/api/data/locator/equals/mesh?data=resources'

    start = 1
    while True:
        req_url = f'{base_url}&batchStart={start}'
        resp = requests.get(req_url)
        resp.raise_for_status()         # Verify 200-response.

        resp_json = resp.json()
        for record in resp_json['results']:
            cip_id = record['id']

            for src_group in record['resources']:
                if src_group['t'] != 632:   # TypeID for MeSH.
                    continue

                for src in src_group['e']:
                    if src['d'] != 'MeSH':
                        continue

                    term = src['u'].replace('+', ' ').lower()
                    if term not in term2cipid_map:
                        term2cipid_map[term] = cip_id
                    break

        end, total = resp_json['end'], resp_json['total']
        print(f"MeSH term -> CIP ID: {end}/{total}", end='\r')

        if end >= total:
            print("MeSH term -> CIP ID: Ok")
            break
        start = end + 1

    # CIP ID --[ChemIDplus]--> InchiKey & SMILES
    cipid2struct_map = {
        'inchikey': {'abbrev': 'ik', 'data': {}},
        'smiles':   {'abbrev': 's',  'data': {}},
    }

    for idx, struct_notation in enumerate(cipid2struct_map):
        base_url = f'https://chem.nlm.nih.gov/api/data/locator/equals/mesh?data={struct_notation}'
        key_abbrev = cipid2struct_map[struct_notation]['abbrev']

        start = 1
        while True:
            req_url = f'{base_url}&batchStart={start}'
            resp = requests.get(req_url)
            resp.raise_for_status()

            resp_json = resp.json()
            for record in resp_json['results']:
                cip_id = record['id']

                rec_data = record.get('structureDetails', {}).get(key_abbrev)
                if rec_data is None:
                    continue
                cipid2struct_map[struct_notation]['data'][cip_id] = rec_data

            end, total = resp_json['end'], resp_json['total']
            print(f"[{idx+1}/2] CIP ID -> {struct_notation}: {end}/{total}", end='\r')

            if end >= total:
                print(f"[{idx+1}/2] CIP ID -> {struct_notation}: Ok")
                break
            start = end + 1

    # Link dicts to map: MeSH ID --> InchiKey & SMILES
    term2struct_map = {term: {} for term in term2cipid_map}
    for term, cip_id in term2cipid_map.items():
        for struct, info in cipid2struct_map.items():
            rec_data = info['data'].get(cip_id)
            term2struct_map[term][struct] = rec_data

    # This implementation gives, implicitly, priority to preferred concepts and terms.
    mesh2struct_map = {struct: {} for struct in cipid2struct_map}
    for struct in mesh2struct_map:
        mapped_ids = set()
        for term, mesh in term2mesh_map.items():
            if mesh in mapped_ids:
                continue

            rec_data = term2struct_map.get(term, {}).get(struct)
            if rec_data is not None:
                mesh2struct_map[struct][mesh] = rec_data
                mapped_ids.add(mesh)

    return mesh2struct_map


class Embeddings:
    def __init__(self, concept):
        self.concept = concept      # Pubtator concept (Chemical, Disease) used to calculate avg document vectors.
        self.mesh2vec_dict = {}

    def get_mean_vecs(self, docs2annots_dict, normalise=False):
        meanvec_dict = {}

        for cc_space, mesh2vec_df in self.mesh2vec_dict.items():
            v_dim = mesh2vec_df.layer_shapes[1]    # Get the dimension of the space's embeddings.
            meanvec_dict_tmp = {}

            # Input given as: 'MESH:#######'.
            known_chemicals = set([f'MESH:{mesh}' for mesh in mesh2vec_df.index.to_list()])

            for doc_idx, annot_freq in docs2annots_dict.items():
                annot_freq = annot_freq[self.concept]
                annot_freq = {m: f for m, f in annot_freq.items() if m in known_chemicals}

                if not annot_freq:
                    meanvec_dict_tmp[doc_idx] = np.zeros(v_dim)
                    continue

                mesh_ids = [mesh.split(':')[1] for mesh in annot_freq]  # Only keep the ID.
                freqs = np.fromiter(annot_freq.values(), dtype=float)
                freqs = freqs / np.sum(freqs)   # After removing terms, re-normalise freqs.

                mesh2vec_slice = mesh2vec_df.loc[mesh_ids]
                if normalise:
                    mesh2vec_slice = mesh2vec_slice.divide(mesh2vec_slice.apply(np.linalg.norm, axis=1), axis=0)

                # Calculate weighted average.
                mean_vec = mesh2vec_slice.multiply(freqs, axis=0).apply(np.mean).to_numpy()
                meanvec_dict_tmp[doc_idx] = mean_vec

            meanvec_dict[cc_space] = meanvec_dict_tmp
        return meanvec_dict


class ChemicalChecker(Embeddings, utils.PickleJar):
    def __init__(self, chemical_mesh_ids):
        super().__init__(concept='Chemical')

        self.basepath = utils.get_data_path('embedded')/'chemical_checker_sign2'    # TODO: Make it downloadable.

        self.cc_spaces_sign2 = []
        for category in ('A', 'B', 'C', 'D', 'E'):
            for level in range(5):
                self.cc_spaces_sign2.append(f'{category}{level+1}')

        mesh_ids = tuple(set(utils.conv2tuple(chemical_mesh_ids)))    # Drop duplicates.

        # CC needs InchiKeys and SMILES. Two alternative sources: ChemIDplus & DrugBank.
        mesh2structure_map = get_mesh2structure_map()
        drugbank = net_utils.DrugBankUtils()
        mesh2structure_map['inchikey'].update(drugbank.mesh2inchikey(mesh_ids))    # Give preference to DrugBank.

        self.chemicals = {'mesh': mesh_ids, 'mesh2structure_map': mesh2structure_map, }

    def populate(self):
        self._import_proc()

    def _import_proc(self):
        mesh2inchikey_map = self.chemicals['mesh2structure_map']['inchikey']
        inchi_keys = set(mesh2inchikey_map.values())      # Membership testing will probably be faster with sets.
        inchikey2mesh_map = {inchi: mesh for mesh, inchi in mesh2inchikey_map.items()}  # Create reverse mapping.

        for cc_space in self.cc_spaces_sign2:
            inchi2vec_dict = {}
            cc_path = self.basepath/f'{cc_space}_sign2.h5'

            with h5py.File(cc_path, 'r') as h5:
                for v_idx, v_key in enumerate(h5['keys']):
                    v_key = v_key.decode('utf-8')   # Originally saved as byte string.

                    if v_key in inchi_keys:
                        cc_vector = h5['V'][v_idx]
                        inchi2vec_dict[v_key] = cc_vector

            mesh2vec_dict_tmp = {inchikey2mesh_map[inchi]: cc_vector for inchi, cc_vector in inchi2vec_dict.items()}

            # Use CC's Signaturizer to fill in missing embeddings.
            cc_sign = Signaturizer(cc_space)

            missing_ids = tuple(set(self.chemicals['mesh']).difference(set(mesh2vec_dict_tmp)))   # Preserve order.
            missing_smiles = [self.chemicals['mesh2structure_map']['smiles'].get(mesh, '') for mesh in missing_ids]

            sign_results = cc_sign.predict(missing_smiles)
            mesh_ids = np.array(missing_ids)[~sign_results.failed]
            cc_vecs = sign_results.signature[~sign_results.failed]

            mesh2vec_dict_tmp.update(dict(zip(mesh_ids, cc_vecs)))

            v_dims = np.fromiter(map(lambda x: 'V'+str(x), range(0, 128)), dtype='<U4')
            self.mesh2vec_dict[cc_space] = pd.DataFrame.from_dict(mesh2vec_dict_tmp, orient='index', columns=v_dims)


class ChemDisMeshTree(Embeddings, utils.PickleJar):
    def __init__(self, concept='Disease', chemdis_mesh_ids=None, build_part=False):
        super().__init__(concept=concept)

        self.basepath = utils.get_data_path('embedded')/'xmlmesh_2022'   # TODO: Make it downloadable.
        self.data = {'mesh': chemdis_mesh_ids, 'concept': concept, 'is_part': build_part, }

    def populate(self):
        self._import_proc()

    # def get_mean_vecs(self, docs2annots_dict, normalise=False):    # Changes default to False.
    #     super().get_mean_vecs(docs2annots_dict, normalise=normalise)

    def _import_proc(self, use_node2vec=True):
        self._build_trees()    # Import and process MeSH descriptor and supplementary data XMLs.
        self._apply_tfidf()    # Vectorise with TF-IDF, and then apply LCA to reduce dimensions.

        if not use_node2vec:
            # Turn to dict and populate supplementary MeSH IDs.
            keys, values = self.data['tfidf'].index.to_list(), self.data['tfidf'].to_numpy()
            mesh2vec_dict = dict(zip(keys, values))
            mesh2vec_dict = self._populate_supp(mesh2vec_dict)

            # Transform back to DataFrame and save the results.
            v_dims = np.fromiter(map(lambda x: 'V' + str(x), range(values.layer_shapes[1])), dtype='<U5')
            self.mesh2vec_dict[self.concept] = pd.DataFrame.from_dict(mesh2vec_dict, orient='index', columns=v_dims)
            return

        self._approx_null()    # Approximate null distribution (1e5 permutations).
        self._create_nets()    # Build undirected similarity graphs (100 neighbours, p-val<.01).
        self._node2vec()       # Calculate the final embeddings using node2vec (128 dimensions).

    @staticmethod
    def _get_concepts(rec):
        out_concepts = {'name': [], 'renu': [], 'terms': [], 'pref_idx': None, }

        concepts = rec.get('ConceptList', {}).get('Concept')
        if concepts is not None:
            if isinstance(concepts, dict):  # It's easier to work with list of concepts (dicts).
                concepts = [concepts]

            for idx, concept in enumerate(list(concepts)):
                is_preferred = concept['@PreferredConceptYN']
                if is_preferred == 'Y':
                    out_concepts['pref_idx'] = idx

                name, renu = concept.get('ConceptName', {}).get('String'), concept.get('RegistryNumber')
                out_concepts['name'].append(name)
                out_concepts['renu'].append(renu)

                terms = concept.get('TermList', {}).get('Term')
                if isinstance(terms, dict):
                    terms = [terms]

                out_terms = []
                for term in terms:
                    out_terms.append(term.get('String'))
                out_concepts['terms'].append(out_terms)

        return out_concepts

    def _build_trees(self, append_concepts=False):
        root_map = {'Disease': 'C', 'Chemical': 'D'}    # Top (root-level) MeSH tree categories.
        target_root = root_map[self.data['concept']]

        # Import descriptor trees.
        desc_path = self.basepath/'desc2022.xml'
        with open(desc_path, 'rb') as f_xml:
            desc_dict_raw = xmltodict.parse(f_xml, process_namespaces=True)

        desc2tree_map = {}
        for desc in desc_dict_raw['DescriptorRecordSet']['DescriptorRecord']:
            trees = desc.get('TreeNumberList', {}).get('TreeNumber')
            if trees is None:
                continue    # In case no tree information is provided, skip to next descriptor.

            trees = utils.conv2tuple(trees)     # Either str or list of strs; convert to tuple.
            trees = set([tree for tree in trees if tree[0] == target_root])
            if not trees:
                continue

            mesh, name = desc.get('DescriptorUI'), desc.get('DescriptorName', {}).get('String')
            if mesh is not None:
                desc2tree_map[mesh] = {'name': name, 'trees': trees, }

                # Append concepts with names & registry numbers. Used to translate to other IDs.
                if append_concepts:
                    desc2tree_map[mesh]['concepts'] = self._get_concepts(desc)

        del desc_dict_raw

        # Read supplementary data.
        supp_map = {'Disease': '3', 'Chemical': '1'}    # Concept to MeSH SCR Class number map.
        target_class = supp_map[self.data['concept']]

        supp_path = self.basepath/'supp2022.xml'
        with open(supp_path, 'rb') as f_xml:
            supp_dict_raw = xmltodict.parse(f_xml, process_namespaces=True)

        supp2desc_map = {}
        for supp in supp_dict_raw['SupplementalRecordSet']['SupplementalRecord']:
            if supp['@SCRClass'] != target_class:
                continue    # Skip if not the target class (Disease or Chemical).

            mappings = supp.get('HeadingMappedToList', {}).get('HeadingMappedTo')
            if mappings is None:    # This should be redundant, unless something's gone wrong.
                continue

            if isinstance(mappings, dict):
                mappings = [mappings, ]  # Can be a dict or list of dicts; keep it consistent.

            desc_set_tmp = set()
            for desc in mappings:
                desc = desc.get('DescriptorReferredTo', {}).get('DescriptorUI')
                if desc is not None:
                    desc = desc[1:]      # Descriptors as: '*D######'; remove the leading '*'.
                    desc_set_tmp.add(desc)

            if desc_set_tmp:
                mesh, name = supp.get('SupplementalRecordUI'), supp.get('SupplementalRecordName', {}).get('String')
                if mesh is not None:
                    supp2desc_map[mesh] = {'name': name, 'descs': desc_set_tmp, }

                    if append_concepts:
                        supp2desc_map[mesh]['concepts'] = self._get_concepts(supp)

        del supp_dict_raw

        # Subset, if requested so.
        if self.data['mesh'] is not None \
                and self.data['is_part']:
            desc2tree_map = {desc: tree for desc, tree in desc2tree_map.items() if desc in self.data['mesh']}
            supp2desc_map = {supp: desc for supp, desc in supp2desc_map.items() if supp in self.data['mesh']}

        # Save all as object data.
        self.data.update({'desc_tree': desc2tree_map, 'supp_desc': supp2desc_map})
        return desc2tree_map, supp2desc_map

    # Get processed MeSH data for external use. By default, also return concepts.
    def get_trees(self, append_concepts=True):
        return self._build_trees(append_concepts=append_concepts)

    def _get_corpus(self):
        corpus_dict = {}
        for mesh, data in self.data['desc_tree'].items():
            trees = tuple(map(lambda t: t.split('.'), data['trees']))

            branches = set()       # Prune any duplicate branches.
            for tree in trees:
                leaf_seq = ''
                for leaf in tree:
                    leaf_seq += f'_{leaf}'
                    branches.add(leaf_seq)

            corpus_dict[mesh] = ' '.join(branches)
        corpus = tuple(corpus_dict.values())
        return corpus_dict, corpus

    def _apply_tfidf(self, lsa=True, center=False, min_df=5, max_df=0.8, svd_comp=0.75, explvar_cutoff=0.9, seed=37):
        corpus_dict, corpus = self._get_corpus()

        # Apply TF-IDF.
        vectoriser = TfidfVectorizer(min_df=min_df, max_df=max_df)
        tfidf_vectors = vectoriser.fit_transform(corpus)

        # Remove all-zero rows. These cause problems later on, but there is also no reason to keep them around.
        nnz_row_mask = tfidf_vectors.getnnz(1) > 0
        tfidf_vectors = tfidf_vectors[nnz_row_mask]

        # The changes need to be reflected on the corpus dict, too.
        nnz_row_indices = np.where(nnz_row_mask)[0]
        for idx, (mesh, _) in enumerate(dict(corpus_dict).items()):    # Iterate over a copy.
            if idx not in nnz_row_indices:
                del corpus_dict[mesh]

        if lsa:
            if center:
                tfidf_vectors = tfidf_vectors.toarray()     # Sparse to dense matrix-- this can explode memory!

                # Normalise to unity. Then, squared euclidean distances are proportional to cosine distances. Mean-
                # centering preserves euclidean, but not cosine distances. To simplify, we'll work with the former.
                normalizer = Normalizer()
                tfidf_vectors = normalizer.transform(tfidf_vectors)
                scaler = StandardScaler(with_std=False)
                tfidf_vectors = scaler.fit_transform(tfidf_vectors)

            _, num_feat = tfidf_vectors.layer_shapes
            num_comp = int(svd_comp*num_feat)   # Number of SVD components.
            trunc_svd = TruncatedSVD(n_components=num_comp, random_state=seed)
            reduced_vectors = trunc_svd.fit_transform(tfidf_vectors)

            # Add a small (enough) number to the reduced vector matrix. This will shift any zero vectors slightly,
            # which is desirable as, otherwise, calculating cosine distance will raise a warning, but also return
            # a distance value of zero (or similarity of 1), messing up calculations. This should solve the issue.
            # Later note: this has been addressed by completely removing all-zero vectors :)
            # reduced_vectors += 1e-6
            expl_var_cumsum = np.cumsum(trunc_svd.explained_variance_ratio_)

            opt_dim = np.amin(np.argwhere(expl_var_cumsum >= explvar_cutoff))
            if not opt_dim:
                warnings.warn(f"Did not meet the explained variance cutoff. Achieved: {expl_var_cumsum[-1]}.")
                opt_dim = num_comp

            tfidf_vectors = reduced_vectors[:, :opt_dim]

        _, num_feat = tfidf_vectors.layer_shapes
        v_dims = np.fromiter(map(lambda x: 'V' + str(x), range(0, num_feat)), dtype='<U5')
        tfidf_vec_df = pd.DataFrame(data=tfidf_vectors, index=corpus_dict, columns=v_dims)
        self.data['tfidf'] = tfidf_vec_df

    @staticmethod
    def _dist_cosine(*vecs):
        return distance.sqeuclidean(*vecs) / 2

    def _approx_null(self, iterations=int(1e5), seed=37):
        num_feat = self.data['tfidf'].layer_shapes[1]
        rng = np.random.default_rng(seed=seed)

        null = np.zeros(iterations, dtype=float)
        for idx in range(iterations):
            print(f"Approximating null distribution. Running permutation: #{idx+1}/{iterations}", end='\r')

            # Pairs are sampled *with* replacement. For each pair, vectors are sampled without replacement.
            pair_idx = rng.choice(num_feat, size=2, replace=False)
            vecs = self.data['tfidf'].iloc[pair_idx, :].to_numpy()

            cos_dist = distance.cosine(*vecs)
            null[idx] = 1 - cos_dist    # Save cosine similarities.

        self.data['null'] = null

    def _create_nets(self, near_neigh=101, min_neigh=3, pval_cutoff=1e-2):
        # Find nearest neighbours.
        vecs = self.data['tfidf'].to_numpy(dtype='float32')

        _, num_feat = vecs.layer_shapes

        # index = faiss.IndexFlatL2(num_feat)     # Squared euclidean distance as metric.
        # index.add(vecs)
        #
        # sim_mat, neigh_mat = index.search(vecs, near_neigh)    # Euclidean distances and neighbour indices.
        # sim_mat = 1 - sim_mat/2                                # Cosine similarity from euclidean distance.

        index = faiss.IndexFlatIP(num_feat)
        faiss.normalize_L2(vecs)   # For cosine similarity, vectors have to be (pre-)normalised to unity.
        index.add(vecs)

        sim_mat, neigh_mat = index.search(vecs, near_neigh)  # Cosine similarities and neighbour indices.
        np.clip(sim_mat, -1, 1, out=sim_mat)

        # Get empirical log-pvals.
        null_nelems = self.data['null'].size

        @np.vectorize
        def calc_logpvals(sim):
            count_ge = np.count_nonzero(self.data['null'] >= sim)
            neglog_pval = -np.log10((1+count_ge)/(1+null_nelems))
            return neglog_pval

        logpval_mat = calc_logpvals(sim_mat)
        np.clip(logpval_mat, -np.log10(5e-2), -np.log10(1e-6), out=logpval_mat)
        min_logpval = -np.log10(pval_cutoff)

        # Build undirected network.
        sim_net = nx.Graph()
        for row_idx, neighbours in enumerate(neigh_mat):
            num_neigh = 0
            for col_idx, neighbour_idx in enumerate(neighbours):
                # Row indices == vector indices. That does not hold for columns. To map, use 'neigh_mat'.
                if row_idx == neighbour_idx:     # Every node has itself as nearest neighbour. Skip that.
                    continue

                # Add the minimum amount of neighbours first. Edge weights don't matter.
                weight = logpval_mat[row_idx, col_idx]
                if num_neigh < min_neigh:
                    sim_net.add_edge(row_idx, neighbour_idx, weight=weight)

                    num_neigh += 1
                    continue

                # Past the required number of neighbours, add edges above the threshold.
                if weight >= min_logpval:
                    sim_net.add_edge(row_idx, neighbour_idx, weight=weight)
                    continue

                break

        # Scale weights in range [1e-2, 1]
        weights = np.fromiter(nx.get_edge_attributes(sim_net, 'weight').values(), dtype=float)
        scaler = MinMaxScaler(feature_range=(1e-2, 1))
        scaler.fit(weights.reshape(-1, 1))  # Reshape because we have one feature.

        for (source, target, weight) in sim_net.edges.data('weight'):
            norm_w = scaler.transform([[weight]])[0][0]
            sim_net[source][target]['weight'] = norm_w

        if not nx.is_connected(sim_net):
            warnings.warn("The similarity graph that was built is not connected.")
        self.data['sim_net'] = sim_net

    def _node2vec(self, dim=128, window=10, **kwargs):
        # Run Node2Vec with default params:
        # Embedding dim= 128, p= 1, q= 1, Length of walk= 80, Walks per source= 10, Context size= 10
        node2vec = Node2Vec(self.data['sim_net'], dimensions=dim, **kwargs)
        # noinspection PyTypeChecker
        node2vec_model = node2vec.fit(window=window)

        # Construct the Index to MeSH map.
        idx2key_map = np.array(node2vec_model.wv.index_to_key, dtype=int)      # Maps node2vec idx to internal IDs.
        key2mesh_map = self.data['tfidf'].index.to_numpy()                     # Maps internal indices to MeSH IDs.
        idx_range = np.arange(idx2key_map.size, dtype=int)
        mesh_dims = [key2mesh_map[idx2key_map[idx]] for idx in idx_range]      # MeSH IDs as they show up in array.

        # Construct the MeSH to vector mapping.
        # Start with the standard descriptors.
        vec_ndarray = node2vec_model.wv.vectors
        mesh2vec_dict = {}
        for idx, vec in enumerate(vec_ndarray):
            mesh = mesh_dims[idx]
            mesh2vec_dict[mesh] = vec

        # # Alternative implementation: this uses the SNAP (C++) implementation of node2vec instead.
        # node2vec_rawdict = utils.node2vec_snap_wrapper(self.data['sim_net'])
        # node2mesh_map = self.data['tfidf'].index.to_numpy()              # Maps nodes to MeSH IDs.
        #
        # mesh2vec_dict = {}
        # for node, vec in node2vec_rawdict.items():
        #     mesh = node2mesh_map[int(node)]
        #     mesh2vec_dict[mesh] = vec

        # Continue with the supplementary ones. To generalise the applicability of the method (as we're inheriting
        # from this class), check and only run this part of code when the appropriate key exists in the dictionary.
        mesh2vec_dict = self._populate_supp(mesh2vec_dict)

        # Finalise structure, then save results.
        v_dims = np.fromiter(map(lambda x: 'V' + str(x), range(dim)), dtype='<U5')
        self.mesh2vec_dict[self.concept] = pd.DataFrame.from_dict(mesh2vec_dict, orient='index', columns=v_dims)
        # self.data['node2vec'] = node2vec_model

    def _populate_supp(self, mesh2vec_dict):
        mesh2vec_dict = dict(mesh2vec_dict)
        if self.data.get('supp_desc'):
            for supp_mesh, mapped_descs in self.data['supp_desc'].items():
                desc_list = mapped_descs['descs']

                vecs_tmp = []
                for desc in desc_list:
                    vec = mesh2vec_dict.get(desc)
                    if vec is not None:
                        vecs_tmp.append(vec)

                if vecs_tmp:
                    vecs_tmp = np.vstack(vecs_tmp)
                    mean_vec = np.mean(vecs_tmp, axis=0)

                    mesh2vec_dict[supp_mesh] = mean_vec     # Append the supplementary MeSH vector to temp dict.
        return mesh2vec_dict


class ChemStringGraph(ChemDisMeshTree):
    def __init__(self, chemical_mesh_ids):
        mesh_ids = tuple(set(utils.conv2tuple(chemical_mesh_ids)))

        # TODO: For now, only partial building is supported. Give option to use all drug MESH IDs available.
        super().__init__(concept='Chemical', chemdis_mesh_ids=mesh_ids, build_part=True)
        del self.basepath

    def _import_proc(self, use_node2vec=True):
        self._build_graph()
        self._apply_tfidf()

        if not use_node2vec:
            self.mesh2vec_dict[self.concept] = self.data['tfidf']
            return

        self._approx_null()
        self._create_nets()
        # self._create_nets(min_neigh=10)  # Changing min. neighbours to 10 creates a fully connected graph.
        self._node2vec()

    def _build_graph(self):
        drugbank = net_utils.DrugBankUtils()
        mesh2targets_map = drugbank.mesh2targets(self.data['mesh'])
        mesh2targets = {m: list(map(lambda x: x[1], trg['targets'])) for m, trg in mesh2targets_map.items()}
        self.data['mesh2trg'] = mesh2targets

    def _get_corpus(self):
        corpus_dict = {}
        for mesh, targets in self.data['mesh2trg'].items():
            corpus_dict[mesh] = ' '.join(targets)

        corpus = tuple(corpus_dict.values())
        return corpus_dict, corpus


class DisCTDBaseGraph(ChemDisMeshTree):
    def __init__(self, disease_mesh_ids):
        mesh_ids = tuple(set(utils.conv2tuple(disease_mesh_ids)))

        super().__init__(concept='Disease', chemdis_mesh_ids=mesh_ids, build_part=True)
        self.basepath = utils.get_data_path('embedded')/'ctdbase'

        self.space_fname_map = {
            'Genes':     'CTD_genes_diseases.csv.gz',
            'Chemicals': 'CTD_chemicals_diseases.csv.gz',
            'Pathways':  'CTD_diseases_pathways.csv.gz',
            'GO-BP':     'CTD_Phenotype-Disease_biological_process_associations.csv.gz',
            'GO-CC':     'CTD_Phenotype-Disease_cellular_component_associations.csv.gz',
            'GO-MF':     'CTD_Phenotype-Disease_molecular_function_associations.csv.gz',
        }
        self.space_dfcol_map = {
            'Genes':     {'DisID': 3, 'TrgID': 1, 'DirEvid': 4   },
            'Chemicals': {'DisID': 4, 'TrgID': 1, 'DirEvid': 5   },
            'Pathways':  {'DisID': 1, 'TrgID': 3, 'DirEvid': None},
            'GO-BP':     {'DisID': 3, 'TrgID': 1, 'DirEvid': None},
            'GO-CC':     {'DisID': 3, 'TrgID': 1, 'DirEvid': None},
            'GO-MF':     {'DisID': 3, 'TrgID': 1, 'DirEvid': None},
        }

    def _import_proc(self, use_node2vec=True):
        for space, fname in self.space_fname_map.items():
            # Change concept (temporarily) to space name. The saving slot is based on that. When we are done,
            # this *has* to be changed back to 'Disease', otherwise mean vector calculation is going to fail.
            self.concept = space         # TODO: Clean this up. Works, but a proper implementation is better.
            ctd_path = self.basepath / fname

            self._build_graph(fpath=ctd_path)
            self._apply_tfidf()

            if not use_node2vec:
                self.mesh2vec_dict[self.concept] = self.data['tfidf']
                return

            self._approx_null()
            self._create_nets()
            self._node2vec()

        self.concept = 'Disease'      # Revert to the correct concept.

    def _build_graph(self, fpath):
        df_edges = pd.read_csv(fpath, skiprows=29, header=None, dtype='string')
        df_cols = dict(self.space_dfcol_map[self.concept])  # Which columns to use for Disease & Target IDs.

        col_direvid = df_cols.pop('DirEvid')
        if col_direvid is not None:
            df_edges = df_edges[~df_edges.iloc[:, col_direvid].isna()]

        cols_keep = list(df_cols.values())
        df_edges = df_edges.iloc[:, cols_keep].copy().drop_duplicates()
        df_edges.iloc[:, 0] = df_edges.iloc[:, 0].apply(lambda mesh: mesh.split(':')[1])
        df_edges.set_index(df_cols['DisID'], inplace=True)

        if self.data['is_part']:
            df_edges = df_edges.loc[df_edges.index.isin(self.data['mesh'])]

        mesh2trgid = {mesh: [] for mesh in df_edges.index}
        for mesh, trg_id in df_edges.iloc[:, 0].items():
            mesh2trgid[mesh].append(trg_id)

        self.data['mesh2trg'] = mesh2trgid

    def _get_corpus(self):
        corpus_dict = {}
        for mesh, targets in self.data['mesh2trg'].items():
            corpus_dict[mesh] = ' '.join(targets)

        corpus = tuple(corpus_dict.values())
        return corpus_dict, corpus


class MeanVectors(Embeddings, utils.PickleJar):
    def __init__(self, concept, mesh_ids):
        super().__init__(concept=concept)
        self.mesh_ids = mesh_ids

        # Pickled drug & disease embeddings calculated with option coarse=False.
        self.basepath = utils.get_data_path('embedded')/'concept_embeddings.pkl'
        with open(self.basepath, 'rb') as pkl:
            self.embeddings = pickle.load(pkl)

    def populate(self, normalise=False, scale_mean=False, scale_std=False, seed=37):
        normaliser = Normalizer()
        scaler = StandardScaler(with_mean=scale_mean, with_std=scale_std)

        # Extract MESH embeddings for concept.
        mesh_df = self.embeddings['MESH'][self.concept]
        # Purge unused MeSH IDs at this point.
        mesh_df = mesh_df[mesh_df.index.isin(self.mesh_ids)]

        # Normalise & scale before trunc. SVD.
        mesh_np = mesh_df.to_numpy()
        if normalise:
            mesh_np = normaliser.fit_transform(mesh_np)
        if scale_mean or scale_std:
            mesh_np = scaler.fit_transform(mesh_np)

        if self.concept == 'Chemical':
            nc = 35     # Explains approx. 80% of the variance.
        else:
            nc = 47     # Explains approx. 90% of the variance.
        trunc_svd = TruncatedSVD(n_components=nc, random_state=seed)
        red_mesh_np = trunc_svd.fit_transform(mesh_np)
        red_mesh_df = pd.DataFrame(red_mesh_np, index=mesh_df.index)

        # Outer join for remaining embeddings.
        concat_df = None
        if self.concept == 'Chemical':
            for _, vecs_df in self.embeddings['CC'].items():
                if concat_df is None:
                    concat_df = vecs_df
                    continue
                concat_df = pd.concat([concat_df, vecs_df], axis=1, join='outer')
            # Drop MeSH IDs that do not show up in text.
            concat_df = concat_df[concat_df.index.isin(self.mesh_ids)]
        else:
            # For CTD, concatenate the 2 largest spaces.
            space_gobp = self.embeddings['CTDBase']['GO-BP']
            space_gomf = self.embeddings['CTDBase']['GO-MF']
            concat_df = pd.concat([space_gobp, space_gomf], axis=1, join='outer')
        concat_df.fillna(0, inplace=True)

        concat_np = concat_df.to_numpy()
        if normalise:
            concat_np = normaliser.fit_transform(concat_np)
        if scale_mean or scale_std:
            concat_np = scaler.fit_transform(concat_np)

        if self.concept == 'Chemical':
            nc = 300-35  # Explains approx. 80% of the variance.
        else:
            nc = 150-47  # Explains approx. 90% of the variance.
        trunc_svd = TruncatedSVD(n_components=nc, random_state=seed)
        red_concat_np = trunc_svd.fit_transform(concat_np)
        red_concat_df = pd.DataFrame(red_concat_np, index=concat_df.index)

        red_space = pd.concat([red_concat_df, red_mesh_df], axis=1, join='outer')
        red_space.fillna(0, inplace=True)
        self.mesh2vec_dict[self.concept] = red_space


if __name__ == '__main__':
    pass

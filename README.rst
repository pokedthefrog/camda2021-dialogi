*dialogi*: Text mining for DILI
===============================

Katritsis, N. M., Liu, A., Youssef, G., Rathee, S., MacMahon, M., Hwang, W., et al. (2022). `dialogi: Utilising NLP
with chemical and disease similarities to drive the identification of Drug-Induced Liver Injury literature`__. bioRxiv,
2022.03.11.483929. `doi:10.1101/2022.03.11.483929`__.

__ https://www.biorxiv.org/content/10.1101/2022.03.11.483929v1
__ https://doi.org/10.1101/2022.03.11.483929

This repository provides preliminary, and currently undocumented, code, alongside the datasets needed to replicate the
results presented in the aforementioned preprint. No plotting functionality is included, therefore this code cannot be
used to reproduce the exact figures shown in the manuscript. Some pre-computed data is uploaded on the repo. To run the
complete pipeline from scratch, additional datasets are required. These are publicly available but– due to their large
size– not uploaded here. Instead, download links and instructions are provided below.

Running *dialogi* with pre-computed data
----------------------------------------

To train, validate, and test the classifiers in a Nested Cross-Validation (NCV) scheme, similar to the one used in
the publication– with 10 outer and 5 inner folds– ``public_results/run_dialogi.py`` is provided. This code depends on
pre-computed concept annotations (from `PubTator`_), processed texts (using `Stanza`_), as well as chemical & disease
embeddings (both from the `Chemical Checker`_ and calculated internally). Running the NCV scheme takes about 10 hours
to complete, on a laptop with an NVIDIA GeForce RTX 2070 Max-Q (with 8GB of VRAM).

.. _PubTator: https://www.ncbi.nlm.nih.gov/research/pubtator/
.. _Stanza: https://stanfordnlp.github.io/stanza/
.. _Chemical Checker: https://chemicalchecker.org/

Acquiring and using additional datasets
---------------------------------------

Download links, together with installation paths and comments, when needed, are given below:

.. |nbsp| unicode:: 0xa0
   :trim:

=========================================================== ======================= ======================================================================
Installation path (``parent`` ``folder name``)              Download link(s)        Comments
=========================================================== ======================= ======================================================================
``data/embedded/`` |nbsp| ``chemical_checker_sign2``        `1 – 25`__              —
``data/embedded/`` |nbsp| ``ctdbase``                       1__ 2__ 3__ 4__ 5__ 6__ —
``data/embedded/`` |nbsp| |nbsp| ``glove_wiki_embeddings``  1__                     *Also required with pre-computed data.* Files must be unzipped first.
``data/embedded/`` |nbsp| ``node2vec_snap``                 1__                     Optional. The ``node2vec`` executable should be placed in the folder.
``data/embedded/`` |nbsp| ``stringppi_v11.0``               Automatic               —
``data/embedded/`` |nbsp| |nbsp| ``xmlmesh_2022``           1__ 2__                 —
=========================================================== ======================= ======================================================================

__ https://chemicalchecker.org/downloads/signature2
__ http://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz
__ http://ctdbase.org/reports/CTD_diseases_pathways.csv.gz
__ http://ctdbase.org/reports/CTD_genes_diseases.csv.gz
__ http://ctdbase.org/reports/CTD_Phenotype-Disease_biological_process_associations.csv.gz
__ http://ctdbase.org/reports/CTD_Phenotype-Disease_cellular_component_associations.csv.gz
__ http://ctdbase.org/reports/CTD_Phenotype-Disease_molecular_function_associations.csv.gz
__ https://nlp.stanford.edu/data/glove.6B.zip
__ http://snap.stanford.edu/snap/download.html
__ https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2022.xml
__ https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp2022.xml

|
The code also utilises `DrugBank`_ to translate from MeSH IDs to InChIKeys. This is part of a paid license and cannot
be shared. The ``mesh2inchikey`` method– found in ``dialogi/net_utils.py``, under the ``DrugBankUtils`` class– can be
modified to always return an empty dictionary, thus bypassing any queries to the database:

.. _DrugBank: https://go.drugbank.com/

.. code-block:: python

    def mesh2inchikey(self, mesh_ids, return_name=False):
        mesh2inchikey_map = {}
        return mesh2inchikey_map

|
Then, the code that follows can be used to retrieve annotations, (pre-)process texts, and calculate the embeddings.
The output produced at this stage should match the pre-computed data uploaded on the repo.

.. code-block:: python

    import pickle
    from dialogi import dialogi, utils


    dlg = dialogi.Dialogi()

    # Retrieve PubTator annotations, then save. Make three calls as, sometimes, retrievals fail.
    for num_trial in range(3):
        num_trial += 1
        print(f'Trial {num_trial}')
        dlg.call_pubtator()
    dlg.save_pubtator(sfx='public')

    # Pre-process with Stanza. This does sentence segmentation, lemmatisation, and UPOS tagging.
    dlg.preproc_docs()
    dlg.save_procdocs(sfx='public')

    # By merging the results from the two previous steps, the final processed texts are created.
    dlg.norm_annot_docs()
    dlg.save_proctxts(sfx='public')

    # Calculate embeddings in two steps: first, concept-level embeddings are learnt and pickled.
    conc_vecs = dlg.calc_concvecs(coarse=False)
    pickle_path = utils.get_data_path('embedded')/'concept_embeddings.pkl'
    pickle.dump(conc_vecs, pickle_path, protocol=pickle.HIGHEST_PROTOCOL)

    # Lastly, average (i.e., text-specific) and dimensionally-reduced embeddings are calculated.
    dlg.calc_concvecs(source_concepts=('MeanVecs_Disease', 'MeanVecs_Chemical'))
    dlg.save_concvecs(sfx='public_reduced')

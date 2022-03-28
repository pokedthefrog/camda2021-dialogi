import sqlite3
from contextlib import closing

import psycopg2

from dialogi import string_db
from dialogi import utils


class StringUtils(string_db.StringPPI):
    def get_ulinks(self, ensembl_ids, threshold=400):
        ensembl_ids = utils.conv2tuple(ensembl_ids)

        ensembl_qmarks = ','.join(len(ensembl_ids) * '?')
        params = 2 * list(ensembl_ids) + [threshold]

        sqlite_ulinks = f"""
                SELECT ensembl_id_a, ensembl_id_b, combined_score
                FROM pp_ulinks
                WHERE ensembl_id_a IN ({ensembl_qmarks}) 
                  AND ensembl_id_b IN ({ensembl_qmarks})
                  AND combined_score >= (?)
                """

        db_path = self.sqlite_path
        with closing(sqlite3.connect(db_path)) as conn:  # auto-closes
            with conn:  # auto-commits
                with closing(conn.cursor()) as db_cur:   # auto-closes
                    db_cur.execute(sqlite_ulinks, params)
                    ulinks = db_cur.fetchall()
        return ulinks


class DrugBankUtils:
    def __init__(self, user='dialogi', passwd='pokedthefrog'):
        self.db_params = {'host': 'localhost', 'dbname': 'drugbank', 'user': user, 'password': passwd}

    def _execute(self, postgres_query, input_iterable):
        in_iter = tuple(map(utils.conv2tuple, input_iterable))

        with closing(psycopg2.connect(**self.db_params)) as conn:
            with conn:
                with closing(conn.cursor()) as db_cur:
                    db_cur.execute(postgres_query, in_iter)
                    fetched_results = db_cur.fetchall()

        return fetched_results

    def mesh2targets(self, mesh_ids):
        postgres_targets = """
            SELECT 
                mappings.code, drugs.name, polypeps.gene_name, polypeps.uniprot_id
            FROM 
                drug_mappings AS mappings
                INNER JOIN drugs                 on mappings.drug_id = drugs.id
                INNER JOIN bonds                 on mappings.drug_id = bonds.drug_id
                INNER JOIN bio_entities ents     on bonds.biodb_id = ents.biodb_id
                INNER JOIN polypeptides polypeps on ents.name = polypeps.name
                INNER JOIN organisms    orgs     on polypeps.organism_id = orgs.id
            WHERE
                mappings.code IN %s
                AND bonds.type    = 'TargetBond'
                AND ents.organism = 'Humans' 
                AND orgs.name     = 'Humans';
        """

        mesh_ids = tuple(set(utils.conv2tuple(mesh_ids)))   # Also remove duplicates.
        target_mappings = self._execute(postgres_targets, (mesh_ids,))

        mesh2targets_map = {}
        for target in target_mappings:
            mesh, name, target_data = target[0], target[1], target[2:]

            if mesh not in mesh2targets_map:
                mesh2targets_map[mesh] = {'name': name, 'targets': [], }
            mesh2targets_map[mesh]['targets'].append(target_data)

        return mesh2targets_map

    def mesh2inchikey(self, mesh_ids, return_name=False):
        postgres_inchikey = """
            SELECT
                mappings.code, props.inchikey, drugs.name
            FROM
                drug_mappings AS mappings
                INNER JOIN drug_calculated_properties props on mappings.drug_id = props.drug_id
                INNER JOIN drugs                            on mappings.drug_id = drugs.id
            WHERE
                mappings.code IN %s;
        """

        mesh_ids = tuple(set(utils.conv2tuple(mesh_ids)))
        inchikey_mappings = self._execute(postgres_inchikey, (mesh_ids,))

        mesh2inchikey_map = {}
        for mapping in inchikey_mappings:
            mesh, inchik, name = mapping

            if not return_name:
                mesh2inchikey_map[mesh] = inchik
                continue
            mesh2inchikey_map[mesh] = {'name': name, 'inchikey': inchik, }

        return mesh2inchikey_map


if __name__ == '__main__':
    pass

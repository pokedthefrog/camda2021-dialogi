import gzip
import re
import sqlite3
import urllib.request
from contextlib import closing
from pathlib import Path

import numpy as np

from dialogi import utils

STRINGDB_BASE_URL = 'https://stringdb-static.org/download'
STRINGDB_BASE_FNS = 'links.detailed,aliases,info,homology_best,actions'.split(',')


class StringPPI:
    def __init__(self, version='11.0', tax_id='9606', force_build=False, force_download=False):
        if tax_id is None:
            tax_id = ''
        self.info = {'ver': version, 'tax_id': tax_id}

        self.string_basepath = utils.get_data_path('embedded')/f'stringppi_v{version}'/tax_id
        self.sqlite_path = self.string_basepath/f'stringppi_v{version}_{tax_id}.sqlite3'

        self._import_txtgz(force_build, force_download)  # (Re-)builds the database, if needed.

    def _import_txtgz(self, force_build, force_download):
        # Create base path, if needed.
        Path(self.string_basepath).mkdir(parents=True, exist_ok=True)

        filepaths = self.filepaths
        db_path = self.sqlite_path

        if (not force_build) \
                and db_path.is_file():
            return

        print("Downloading missing STRING data...")
        for fpath in filepaths:
            if (not force_download) \
                    and fpath.exists():
                continue

            url = self._build_url(fpath)
            with utils.DownloadProgressBar(desc=fpath.name) as progress:
                urllib.request.urlretrieve(url, fpath, reporthook=progress.update_to)
        print("Downloading completed successfully.")

        print("\nBuilding database...")
        with closing(sqlite3.connect(db_path)) as conn:    # auto-closes
            with conn:  # auto-commits
                with closing(conn.cursor()) as db_cur:     # auto-closes
                    self._sqlite_create_tables(db_cur)

                    for fpath in filepaths:
                        populate_func = self._get_populate_func(fpath)

                        with gzip.open(fpath, 'rt') as file:
                            file.readline()
                            populate_func(db_cur, file)
                            print("\t\tDone.")

    def _get_populate_func(self, filepath):
        # Must have 1:1 relationship with STRINGDB_BASE_FNS.
        ordered_funcs = (
            self._populate_ppulinks,    # This also populates the 'proteins' table.
            self._populate_aliases,
            self._populate_prefname,
            self._populate_homologs,
            self._populate_ppdlinks,
        )

        func_map = dict(zip(self.filepaths, ordered_funcs))
        return func_map[filepath]

    @property
    def filenames(self):
        ver, tax = self.info.values()

        def build_fn(bf):
            filename = f'protein.{bf}.v{ver}.txt.gz'
            if tax:
                filename = '.'.join([tax, filename])
            return filename

        string_filenames = tuple(map(build_fn, STRINGDB_BASE_FNS))
        return string_filenames

    @property
    def filepaths(self):
        filenames = self.filenames
        base = self.string_basepath
        string_filepaths = tuple([base/file for file in filenames])

        return string_filepaths

    def _build_url(self, filepath):
        url_filepath = filepath.name

        # If specific taxid is requested, a subdir has to be added.
        if self.info['tax_id']:
            subdir = '.'.join(url_filepath.split('.')[1:-2])
            url_filepath = '/'.join([subdir, url_filepath])

        dl_url = '/'.join([STRINGDB_BASE_URL, url_filepath])
        return dl_url

    @staticmethod
    def _sqlite_create_tables(db_cur):
        db_cur.executescript("""
            CREATE TABLE IF NOT EXISTS proteins
            (
                ensembl_id TEXT NOT NULL UNIQUE,
                species    TEXT NOT NULL,
                PRIMARY KEY (ensembl_id)
            );

            CREATE TABLE IF NOT EXISTS pp_ulinks
            (
                ensembl_id_a   TEXT    NOT NULL,
                ensembl_id_b   TEXT    NOT NULL,
                combined_score INTEGER NOT NULL,
                PRIMARY KEY (ensembl_id_a, ensembl_id_b),
                
                FOREIGN KEY (ensembl_id_a)
                    REFERENCES proteins (ensembl_id),
                FOREIGN KEY (ensembl_id_b)
                    REFERENCES proteins (ensembl_id)
            );

            CREATE TABLE IF NOT EXISTS aliases
            (
                ensembl_id TEXT NOT NULL,
                alias      TEXT NOT NULL,
                sources    TEXT NOT NULL,
                PRIMARY KEY (ensembl_id, alias, sources),
                
                FOREIGN KEY (ensembl_id)
                    REFERENCES proteins (ensembl_id)
            );

            CREATE TABLE IF NOT EXISTS pref_name
            (
                ensembl_id TEXT NOT NULL UNIQUE,
                pref_alias TEXT NOT NULL,
                PRIMARY KEY (ensembl_id),
                
                FOREIGN KEY (ensembl_id)
                    REFERENCES proteins (ensembl_id)
            );

            CREATE TABLE IF NOT EXISTS homologs
            (
                ensembl_id_a TEXT    NOT NULL,
                species_a    TEXT    NOT NULL,
                ensembl_id_b TEXT    NOT NULL,
                species_b    TEXT    NOT NULL,
                bit_score    INTEGER NOT NULL,
                PRIMARY KEY (ensembl_id_a, ensembl_id_b),
                
                FOREIGN KEY (ensembl_id_a)
                    REFERENCES proteins (ensembl_id)
            );

            CREATE TABLE IF NOT EXISTS pp_dlinks
            (
                ensembl_id_a   TEXT    NOT NULL,
                ensembl_id_b   TEXT    NOT NULL,
                mode_action    TEXT    NOT NULL,
                directionality TEXT    NOT NULL,
                combined_score INTEGER NOT NULL,
                PRIMARY KEY (ensembl_id_a, ensembl_id_b, mode_action, directionality),
                
                FOREIGN KEY (ensembl_id_a, ensembl_id_b)
                    REFERENCES pp_ulinks (ensembl_id_a, ensembl_id_b)
            );""")

    # Helper method used when filling the tables.
    @staticmethod
    def _parse_line(line, sep=r'\t+'):
        recs = re.split(sep, line)
        recs = np.array([*map(str.rstrip, recs)])
        return recs

    def _populate_ppulinks(self, db_cur, f_obj):
        print("\t[1/5] Populating PP_uLinks...")

        proteins = set()
        for r in f_obj:
            recs = self._parse_line(r, sep=r'\s+')
            proteins.update(recs[0:2])

            _, ensemblid_a = recs[0].split('.')
            _, ensemblid_b = recs[1].split('.')
            combined_score = int(recs[9])

            data = (ensemblid_a, ensemblid_b, combined_score)
            db_cur.execute("INSERT INTO pp_ulinks VALUES (   ?, ?, ?   )", data)

        for protein in proteins:
            tax_id, ensembl_id = protein.split('.')

            data = (ensembl_id, tax_id)
            db_cur.execute("INSERT INTO proteins  VALUES (    ?, ?     )", data)

    def _populate_aliases(self, db_cur, f_obj):
        print("\t[2/5] Populating aliases...")

        for r in f_obj:
            recs = self._parse_line(r)

            _, ensembl_id = recs[0].split('.')
            alias, sources = recs[1:3]

            data = (ensembl_id, alias, sources)
            db_cur.execute("INSERT INTO  aliases  VALUES (   ?, ?, ?   )", data)

    def _populate_prefname(self, db_cur, f_obj):
        print("\t[3/5] Populating pref_names...")

        for r in f_obj:
            recs = self._parse_line(r)

            protein, pref_alias = recs[0:2]
            _, ensembl_id = protein.split('.')

            data = (ensembl_id, pref_alias)
            db_cur.execute("INSERT INTO pref_name VALUES (    ?, ?     )", data)

    def _populate_homologs(self, db_cur, f_obj):
        print("\t[4/5] Populating homologs...")

        for r in f_obj:
            recs = self._parse_line(r)

            taxid_a, ensemblid_a = recs[1].split('.')

            b_tmp = recs[3].split('.')
            taxid_b = b_tmp[0]
            protein_b = '.'.join(b_tmp[1:])
            bit_score = float(recs[4])

            data = (ensemblid_a, taxid_a, protein_b, taxid_b, bit_score)
            db_cur.execute("INSERT INTO homologs  VALUES (?, ?, ?, ?, ?)", data)

    def _populate_ppdlinks(self, db_cur, f_obj):
        print("\t[5/5] Populating PP_dLinks...")

        for r in f_obj:
            recs = self._parse_line(r, sep=r'\t')

            _, ensemblid_a = recs[0].split('.')
            _, ensemblid_b = recs[1].split('.')
            mode_action = '_'.join(filter(None, recs[2:4]))

            direction = 'none'
            if recs[4] == 't':
                if recs[5] == 't':
                    direction = 'a'
                else:
                    direction = 'b'

            score = recs[6]

            data = (ensemblid_a, ensemblid_b, mode_action, direction, score)
            db_cur.execute("INSERT INTO pp_dlinks VALUES (?, ?, ?, ?, ?)", data)


if __name__ == '__main__':
    string = StringPPI()

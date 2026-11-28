"""Check directory/ZIP Kaggle input without downloading data or training."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

NOTEBOOK = Path(__file__).resolve().parents[1]/'notebooks/model_a_reference_pix2pix_kaggle.ipynb'
nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
code = next(c['source'] for c in nb['cells'] if c['id']=='cell-008')


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'images'
        for i in range(8):
            for folder in ['sketch','reference']:
                path = self.data/folder/'000_character'/f'{i}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'image fixture')
        (self.data/'sketch/000_character/unpaired.png').write_bytes(b'unpaired')

    def run_input(self, path):
        run = self.root/'run'
        (run/'splits').mkdir(parents=True, exist_ok=True)
        ns = dict(Path=Path, zipfile=zipfile, hashlib=hashlib, re=re, csv=csv,
                  json=json, random=random, RUN=run,
                  cfg=SimpleNamespace(DATA_ROOT=str(path), WORK_ROOT=str(self.root/'working'), SEED=5413),
                  stable_seed=lambda value:int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:4], 'little'))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code,'notebook-input','exec'),ns)
        return ns

    def test_directory_pairs_and_splits(self):
        ns = self.run_input(self.data)
        self.assertEqual(ns['data_summary']['paired'],8)
        self.assertEqual(ns['data_summary']['sketch_only'],1)
        self.assertEqual(sum(map(len,ns['SPLITS'].values())),8)
        self.assertTrue((self.data/'sketch/000_character/unpaired.png').is_file())

    def test_zip_extracts_once_and_keeps_same_splits(self):
        source = self.run_input(self.data)
        archive = self.root/'AnimeFaceColorization_paired.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for path in self.data.rglob('*.png'):
                z.write(path,path.relative_to(self.data))
        extracted = self.run_input(archive)
        self.assertEqual(source['DATA_SIGNATURE'],extracted['DATA_SIGNATURE'])
        self.assertTrue((extracted['DATA_ROOT']/'.ready').is_file())
        with patch.object(zipfile,'ZipFile',side_effect=AssertionError('ZIP was unnecessarily extracted again')):
            repeat = self.run_input(archive)
        self.assertEqual(extracted['DATA_SIGNATURE'],repeat['DATA_SIGNATURE'])

    def test_bad_input_path_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError,'Kaggle Input'):
            self.run_input(self.root/'not-attached')

    def test_no_hugging_face_auth_or_dataset_download_in_notebook(self):
        source = '\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        for removed in ['HF_TOKEN','DOWNLOAD_DATASET','HF_REVISION','huggingface_hub','snapshot_download']:
            self.assertNotIn(removed,source)


if __name__=='__main__':
    unittest.main()

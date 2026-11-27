import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('packager', Path(__file__).resolve().parents[1]/'scripts/package_paired_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ['sketch/character/a.jpg', 'reference/character/a.jpg',
                     'sketch/character/missing_color.jpg', 'reference/character/missing_sketch.jpg',
                     '.cache/huggingface/download/metadata', 'README.md']:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())

    def test_packages_only_pairs_then_cleans_unpaired_and_cache(self):
        with patch.object(module, 'tqdm', side_effect=lambda items,**kwargs:items):
            archive = module.package_and_clean(self.root)
            # A second run reuses the verified archive without losing paired images.
            module.package_and_clean(self.root)
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(set(z.namelist()), {'sketch/character/a.jpg','reference/character/a.jpg',
                                                'README.md','dataset_summary.json','excluded_files.txt'})
        self.assertTrue((self.root/'sketch/character/a.jpg').is_file())
        self.assertTrue((self.root/'reference/character/a.jpg').is_file())
        self.assertFalse((self.root/'sketch/character/missing_color.jpg').exists())
        self.assertFalse((self.root/'reference/character/missing_sketch.jpg').exists())
        self.assertFalse((self.root/'.cache').exists())

    def test_zip_verification_failure_prevents_deletion(self):
        with patch.object(module, 'tqdm', side_effect=lambda items,**kwargs:items), \
             patch.object(zipfile.ZipFile, 'testzip', return_value='sketch/character/a.jpg'), \
             self.assertRaises(ValueError):
            module.package_and_clean(self.root)
        self.assertTrue((self.root/'sketch/character/missing_color.jpg').exists())
        self.assertTrue((self.root/'reference/character/missing_sketch.jpg').exists())
        self.assertTrue((self.root/'.cache').exists())

    def test_source_changes_prevent_deletion(self):
        def simulate_new_download(zip_file):
            (self.root/'reference/character/missing_color.jpg').write_bytes(b'new target')
            return None
        with patch.object(module, 'tqdm', side_effect=lambda items,**kwargs:items), \
             patch.object(zipfile.ZipFile, 'testzip', simulate_new_download), \
             self.assertRaises(RuntimeError):
            module.package_and_clean(self.root)
        self.assertTrue((self.root/'sketch/character/missing_color.jpg').exists())
        self.assertTrue((self.root/'reference/character/missing_color.jpg').exists())
        self.assertTrue((self.root/'.cache').exists())


if __name__ == '__main__':
    unittest.main()

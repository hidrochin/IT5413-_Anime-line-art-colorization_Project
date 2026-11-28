"""Check the local downloader without network requests or writing to D:."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1]/'scripts/download_dataset.py'
spec = importlib.util.spec_from_file_location('download_dataset', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def test_lists_only_the_two_subfolders_and_preserves_download_location(self):
        api = Mock()
        entries = []
        for folder in ['sketch','reference']:
            image = Mock(spec=module.RepoFile)
            image.path = f'{folder}/character/image.png'
            entries.append([SimpleNamespace(path=f'{folder}/character'), image])
        api.list_repo_tree.side_effect = entries
        calls = []

        def download(repo_id, filename, **kwargs):
            calls.append((repo_id, filename, kwargs))

        def run_threads(function, files, **kwargs):
            return [function(filename) for filename in files]

        with patch.object(module, 'HfApi', return_value=api), \
             patch.object(module, 'hf_hub_download', side_effect=download), \
             patch.object(module, 'thread_map', side_effect=run_threads):
            module.download_dataset(Path('existing-data'), 'hf_exampleToken')

        api.whoami.assert_called_once()
        self.assertEqual([call.kwargs['path_in_repo'] for call in api.list_repo_tree.call_args_list],
                         ['sketch','reference'])
        self.assertEqual([call[1] for call in calls],
                         ['README.md','sketch/character/image.png','reference/character/image.png'])
        self.assertTrue(all(call[2]['local_dir'] == 'existing-data' for call in calls))
        self.assertTrue(all(call[2]['token'] == 'hf_exampleToken' for call in calls))
        self.assertTrue(all('force_download' not in call[2] for call in calls))

    def test_rejects_assignment_instead_of_token_value(self):
        with patch.object(module, 'HfApi') as api, self.assertRaises(SystemExit):
            module.download_dataset(Path('existing-data'), 'HF_TOKEN = "hf_exampleToken"')
        api.return_value.whoami.assert_not_called()

    def test_failed_authentication_stops_before_downloading(self):
        api = Mock()
        api.whoami.side_effect = RuntimeError('authentication failed')
        with patch.object(module, 'HfApi', return_value=api), \
             patch.object(module, 'hf_hub_download') as download, self.assertRaises(RuntimeError):
            module.download_dataset(Path('existing-data'), 'hf_exampleToken')
        download.assert_not_called()


if __name__ == '__main__':
    unittest.main()

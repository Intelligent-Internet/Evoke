from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_preflight as preflight


class DeviceReservationTests(unittest.TestCase):
    def test_cpu_preparation_does_not_request_gpu(self):
        args = SimpleNamespace(mode='witness')
        with patch.object(preflight, 'run_controller', return_value='CPU') as run:
            self.assertEqual(preflight.controller(args), 'CPU')
            run.assert_called_once_with(args)

    def test_cuda_requires_explicit_device(self):
        args = SimpleNamespace(mode='cuda-model', gpu=None)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            preflight.controller(args)

    def test_occupied_gpu_never_launches_or_evicts(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(mode='cuda-model', gpu=0, research_root=Path(temp))
            for state in ('1024, 0', '1, 30'):
                with patch.object(preflight.subprocess, 'check_output', return_value=state), \
                     patch.object(preflight, 'run_controller') as run:
                    with self.assertRaisesRegex(ValueError, 'occupied'):
                        preflight.controller(args)
                    run.assert_not_called()

    def test_idle_gpu_uses_existing_cooperative_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(mode='cuda-model', gpu=0, research_root=Path(temp))
            with patch.object(preflight.subprocess, 'check_output', return_value='1, 0'), \
                 patch.object(preflight, 'run_controller', return_value='checked'):
                self.assertEqual(preflight.controller(args), 'checked')
                self.assertTrue((Path(temp) / '.ng-gpu-0.lock').exists())


if __name__ == '__main__':
    unittest.main()

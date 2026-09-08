"""Regression checks for the Windows file lock that interrupted training."""
import ctypes
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import train_baseline as training


class TrainingFileTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows sharing behavior")
    def test_json_save_recovers_when_reader_releases_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            path.write_text('{"epoch": 54}', encoding="utf-8")
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_void_p,
                                          ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_void_p]
            kernel.CreateFileW.restype = ctypes.c_void_p
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.CloseHandle.restype = ctypes.c_int
            # Allow reads/writes but deny deletion, as some Windows readers do.
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            release = threading.Timer(0.4, kernel.CloseHandle, args=(handle,))
            release.start()
            try:
                training.save_json(path, {"epoch": 55})
            finally:
                release.join()
            self.assertEqual(training.json.loads(path.read_text()), {"epoch": 55})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_nonessential_progress_does_not_stop_training(self):
        with patch.object(training, "save_json", side_effect=PermissionError("locked")):
            training.save_status(Path("status.json"), {"epoch": 55})

    def test_failed_checkpoint_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last.pt"
            path.write_bytes(b"previous complete checkpoint")
            with patch.object(training, "replace_with_retry", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    training.save_checkpoint(path, {"epoch": 55})
            self.assertEqual(path.read_bytes(), b"previous complete checkpoint")
            self.assertTrue(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()

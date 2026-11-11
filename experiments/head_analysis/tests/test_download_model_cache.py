import os
import subprocess
import sys


def test_import_respects_huggingface_environment():
    env = dict(os.environ, HF_ENDPOINT="https://example.invalid", HF_HUB_ETAG_TIMEOUT="19",
               HF_HUB_DOWNLOAD_TIMEOUT="29", HF_HUB_DISABLE_XET="1", HF_HUB_ENABLE_HF_TRANSFER="1")
    code = (
        "import os; import experiments.head_analysis.scripts.download_model_cache; "
        "assert os.environ['HF_ENDPOINT'] == 'https://example.invalid'; "
        "assert os.environ['HF_HUB_ETAG_TIMEOUT'] == '19'; "
        "assert os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] == '29'; "
        "assert os.environ['HF_HUB_DISABLE_XET'] == '1'; "
        "assert os.environ['HF_HUB_ENABLE_HF_TRANSFER'] == '1'; "
        "from huggingface_hub import constants; assert constants.ENDPOINT == 'https://example.invalid'"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

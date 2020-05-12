import subprocess
import glob
import tempfile
import pytest

notebooks = [nb for nb in glob.glob("tests/*.ipynb")]


@pytest.mark.parametrize("nb", notebooks)
def test_execute_notebooks(nb):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook",
                "--execute", "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, nb]

        subprocess.check_call(args)

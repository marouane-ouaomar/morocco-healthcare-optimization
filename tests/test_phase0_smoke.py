"""
Phase 0 smoke tests.

Verifies that the project structure exists and core dependencies import correctly.
Real unit tests are added in Phases 1–6 alongside each module.
"""
import importlib
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


class TestProjectStructure:
    """Verify all required directories and files are present."""

    def test_src_directory_exists(self):
        assert (PROJECT_ROOT / "src").is_dir()

    def test_scripts_directory_exists(self):
        assert (PROJECT_ROOT / "scripts").is_dir()

    def test_data_directory_exists(self):
        assert (PROJECT_ROOT / "data").is_dir()

    def test_docs_directory_exists(self):
        assert (PROJECT_ROOT / "docs").is_dir()

    def test_webapp_directory_exists(self):
        assert (PROJECT_ROOT / "webapp").is_dir()

    def test_requirements_txt_exists(self):
        assert (PROJECT_ROOT / "requirements.txt").is_file()

    def test_env_example_exists(self):
        assert (PROJECT_ROOT / ".env.example").is_file()

    def test_license_exists(self):
        assert (PROJECT_ROOT / "LICENSE").is_file()

    def test_readme_exists(self):
        assert (PROJECT_ROOT / "README.md").is_file()

    def test_readme_has_ethics_section(self):
        readme = (PROJECT_ROOT / "README.md").read_text()
        assert "Ethics" in readme or "ethics" in readme, "README must contain an ethics section"

    def test_env_example_has_no_real_secrets(self):
        """Ensure .env.example contains only placeholder values."""
        content = (PROJECT_ROOT / ".env.example").read_text()
        # Real keys are typically 40+ char alphanumeric strings
        import re
        # Anthropic keys start with sk-ant-, OpenAI with sk-
        assert "sk-ant-" not in content, "Real Anthropic key found in .env.example!"
        assert re.search(r'sk-[a-zA-Z0-9]{40,}', content) is None, \
            "Real API key found in .env.example!"


class TestDependencyImports:
    """Verify core dependencies are importable (i.e., installed correctly)."""

    def test_import_pandas(self):
        assert importlib.import_module("pandas") is not None

    def test_import_numpy(self):
        assert importlib.import_module("numpy") is not None

    def test_import_geopandas(self):
        assert importlib.import_module("geopandas") is not None

    def test_import_scipy(self):
        assert importlib.import_module("scipy") is not None

    def test_import_sklearn(self):
        assert importlib.import_module("sklearn") is not None

    def test_import_requests(self):
        assert importlib.import_module("requests") is not None

    def test_import_dotenv(self):
        assert importlib.import_module("dotenv") is not None

    def test_import_pydantic(self):
        assert importlib.import_module("pydantic") is not None

"""The paper folders of a project branch (papers/<name>/): what one paper needs on top of client-f.

Each folder uses CLiENT's hpc/ tools and keeps no copy of them; every call and import of them
names a file this checkout has; every run and profile YAML loads with CLiENT's own schema; every
shell script parses; env.sh finds this checkout and the folder. The paper's builds and true-
likelihood jobs need the results of the runs and a cluster, and are not checked here.
On client-f itself there is no papers/ folder and nothing is checked."""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HPC = ROOT / "hpc"
PAPERS = sorted(p for p in (ROOT / "papers").glob("*") if p.is_dir()) if (ROOT / "papers").is_dir() else []
# CLiENT's generic cluster tools, by module name, and where in hpc/ they are
TOOLS = {"launch": "launch.py", "domain": "domain.py", "refused": "refused.py", "loading": "loading.py",
         "boltzmann_build": "boltzmann_build.py",
         **{n: f"analysis/{n}.py" for n in ("branch_run", "chain_sampling", "corner", "evidence_summary", "find_centre",
                                            "make_box", "report", "true_check")},
         **{n: f"true_profiles/{n}.py" for n in ("cluster_backend", "launch_profiles", "make_config", "status")}}


def files(folder, suffix):
    return sorted(p for p in folder.rglob(f"*{suffix}") if "__pycache__" not in p.parts)


@unittest.skipUnless(PAPERS, "no papers/ folder: client-f itself")
class PaperFolders(unittest.TestCase):
    def test_no_copy_of_a_client_tool(self):
        for paper in PAPERS:
            copies = [str(p.relative_to(paper)) for p in files(paper, ".py") if p.stem in TOOLS]
            self.assertEqual(copies, [], f"{paper.name}: use CLiENT's (hpc/) and change it there")

    def test_the_tools_resolve_to_client_s(self):
        for paper in PAPERS:
            sys.path.insert(0, str(paper))
            try:
                import client_paths
                self.assertEqual(client_paths.CLIENT_REPO.resolve(), ROOT.resolve())
                for name, path in TOOLS.items():
                    spec = importlib.util.find_spec(name)
                    self.assertEqual(Path(spec.origin).resolve(), (HPC / path).resolve(), name)
            finally:
                sys.path.remove(str(paper))
                sys.modules.pop("client_paths", None)

    def test_every_call_from_a_script_names_a_file_client_has(self):
        named = re.compile(r"\\?\$\{?CLIENT_REPO(?::\?[^}]*)?\}?/hpc/([\w./-]+)")
        for paper in PAPERS:
            missing = [f"{s.relative_to(paper)}: hpc/{path}" for s in files(paper, ".sh")
                       for path in named.findall(s.read_text()) if not (HPC / path).is_file()]
            self.assertEqual(missing, [], paper.name)

    def test_every_program_that_imports_a_tool_does_it_through_client_paths(self):
        imported = re.compile(r"^\s*(?:from|import) (\w+)", re.MULTILINE)
        for paper in PAPERS:
            for program in files(paper, ".py"):
                text = program.read_text()
                if set(imported.findall(text)) & set(TOOLS):
                    self.assertIn("import client_paths", text, f"{paper.name}/{program.relative_to(paper)}")

    def test_every_run_and_profile_yaml_loads_with_client_s_schema(self):
        sys.path.insert(0, str(ROOT))
        from config.config import Config
        from profiles.config import ProfileConfig

        for paper in PAPERS:
            problems, runs = [], 0
            for path in files(paper / "runs", ".yaml"):
                data = yaml.safe_load(path.read_text())
                if not isinstance(data, dict):
                    continue
                try:
                    if path.name == "client.yaml":
                        runs += 1
                        Config.from_dict(data)
                    elif path.name.startswith("profiles"):
                        ProfileConfig(**dict(data, workers=1)).validate()     # workers are the cluster node's, not this host's
                except Exception as error:      # noqa: BLE001 -- every refusal is listed, not only the first
                    problems.append(f"{path.relative_to(paper)}: {error}")
            self.assertGreater(runs, 0, paper.name)
            self.assertEqual(problems, [], paper.name)

    def test_every_shell_script_parses(self):
        for paper in PAPERS:
            problems = []
            for script in files(paper, ".sh"):
                shell = "sh" if script.read_text().startswith("#!/bin/sh") else "bash"
                done = subprocess.run([shell, "-n", str(script)], capture_output=True, text=True)
                if done.returncode:
                    problems.append(f"{script.relative_to(paper)}: {done.stderr.strip()}")
            self.assertEqual(problems, [], paper.name)

    @unittest.skipUnless(shutil.which("bash"), "needs bash")
    def test_env_finds_the_checkout_and_the_folder(self):
        clean = {k: v for k, v in os.environ.items() if k not in ("HPC_CLIENT_DIR", "CLIENT_REPO", "CLIENT_HPC_DIR", "CLASSPP_COMMIT")}
        clean["CONDA_ROOT"] = "/nonexistent"
        for paper in [p for p in PAPERS if (p / "env.sh").exists()]:
            said = subprocess.run(["bash", "-c", 'source env.sh >/dev/null 2>&1; printf "%s\\n" "$CLIENT_REPO" "$CLIENT_HPC_DIR"'],
                                  cwd=paper, env=clean, capture_output=True, text=True, check=True).stdout.split("\n")
            self.assertEqual([Path(s).resolve() for s in said[:2]], [ROOT.resolve(), paper.resolve()], paper.name)


if __name__ == "__main__":
    unittest.main()

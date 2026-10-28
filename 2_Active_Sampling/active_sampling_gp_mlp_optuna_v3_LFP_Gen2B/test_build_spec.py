import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class BuildSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec_path = Path(__file__).with_name("SurrogatePredictor.spec")
        analysis = Mock(return_value=SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[]))
        with patch.object(sys, "argv", [str(spec_path)]):
            cls.spec = runpy.run_path(
                str(spec_path),
                init_globals={"Analysis": analysis, "PYZ": Mock(), "EXE": Mock(), "COLLECT": Mock()},
            )
        cls.analysis_options = analysis.call_args.kwargs
        cls.data_paths = {
            (Path(destination) / Path(source).name).as_posix()
            for source, destination in cls.analysis_options["datas"]
        }

    def test_server_startup_hello_dependency_is_retained(self):
        for module in ["streamlit.hello", "streamlit.hello.streamlit_app"]:
            self.assertIn(module, self.analysis_options["hiddenimports"])
            self.assertFalse(any(
                module == excluded or module.startswith(excluded + ".")
                for excluded in self.analysis_options["excludes"]
            ))
        self.assertIn("streamlit/hello/streamlit_app.py", self.data_paths)

        from streamlit.hello import streamlit_app
        from streamlit.web.server.server import Server

        application = SimpleNamespace(_main_script_path="_st_app_entry.py")
        hello = SimpleNamespace(_main_script_path=streamlit_app.__file__)
        self.assertFalse(Server.is_running_hello.fget(application))
        self.assertTrue(Server.is_running_hello.fget(hello))

    def test_unneeded_resources_remain_excluded(self):
        for prefix in ["streamlit/.agents/", "streamlit/testing/", "streamlit/development/", "outputs/"]:
            self.assertFalse(any(path.startswith(prefix) for path in self.data_paths), prefix)
        for module in ["streamlit.testing", "streamlit.development"]:
            self.assertIn(module, self.analysis_options["excludes"])
            self.assertFalse(any(
                name == module or name.startswith(module + ".")
                for name in self.analysis_options["hiddenimports"]
            ))
        self.assertIn("streamlit/static/index.html", self.data_paths)
        self.assertNotIn("embedded_bundle/latest_surrogate_bundle.pkl", self.data_paths)


if __name__ == "__main__":
    unittest.main()
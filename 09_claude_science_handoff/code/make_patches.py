import re, os, sys, shutil, subprocess
ROOT = "/home/enserrog/ising_EnergyOverTime/energy_over_time_organized"
P = os.path.abspath("patched")
for sub in ["shared", "03_j_coupling_vs_kinematics_timing/code", "04_collective_activity_pk/code"]:
    shutil.copytree(f"{ROOT}/{sub}", f"{P}/{sub}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.png", "*.csv"))


def edit(rel, old, new):
    f = f"{P}/{rel}"
    s = open(f).read()
    assert s.count(old) == 1, (rel, s.count(old), old[:60])
    open(f, "w").write(s.replace(old, new))


# 1. session label in aggregate_transition_analysis.py
OLD1 = '''def session_label(sdir: Path) -> str:
    """Human-readable label: parent folder name (rep) + leaf."""
    return sdir.parent.name
'''
NEW1 = '''def session_label(sdir: Path) -> str:
    """Human-readable label: the nearest ancestor folder named '<experiment>_repN'.

    In the organized layout the CSV sits in <session>_results/<session>_rep1/full_reach/stim_K,
    so sdir.parent.name is 'full_reach' for every session and all sessions collapsed into one
    label. Walk up to the first '_repN' folder instead; fall back to the parent name."""
    import re
    for p in [sdir] + list(sdir.parents):
        if re.search(r"_rep\\d+$", p.name, flags=re.IGNORECASE):
            return p.name
    return sdir.parent.name
'''
edit("shared/code/aggregate_transition_analysis.py", OLD1, NEW1)

# 2. backslash inside f-string expression (needs Python >= 3.12) in pk_j_analysis.py
_f = f"{P}/04_collective_activity_pk/code/pk_j_analysis.py"
_s = open(_f).read()
_n = _s.count("{'MET \\u2713'") + _s.count("{'non-normal \\u2713'")
assert _n == 3, _n
_s = _s.replace("{'MET \\u2713'", "{'MET ' + CHECK").replace("{'non-normal \\u2713'", "{'non-normal ' + CHECK")
_lines = _s.split("\n")
_last_import = max(i for i, l in enumerate(_lines) if l.startswith(("import ", "from ")) and "__future__" not in l)
if "(" in _lines[_last_import] and ")" not in _lines[_last_import]:
    while ")" not in _lines[_last_import]:
        _last_import += 1
_lines.insert(_last_import + 1, "\nCHECK = \"\\u2713\"  # used inside f-string expressions; a literal backslash there needs Python >= 3.12")
open(_f, "w").write("\n".join(_lines))
compile(open(_f).read(), _f, "exec")

# 3. make shared/src and the P(K) module importable without PYTHONPATH
BLOCK = '''
# Make shared/src and the P(K) analysis module importable without PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_PROJ, "shared"), os.path.join(_PROJ, "04_collective_activity_pk", "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

'''
edit("03_j_coupling_vs_kinematics_timing/code/arbitration_j_many.py", "from src.processing import", BLOCK + "from src.processing import")
edit("03_j_coupling_vs_kinematics_timing/code/test_collective_j_timing.py",
     "from src.util import find_file_recursive\nfrom pk_j_analysis",
     BLOCK + "from src.util import find_file_recursive\nfrom pk_j_analysis")

os.makedirs("patches", exist_ok=True)
for rel, name in [("shared/code/aggregate_transition_analysis.py", "01_aggregate_session_label.diff"),
                  ("04_collective_activity_pk/code/pk_j_analysis.py", "02_pk_j_analysis_fstring.diff"),
                  ("03_j_coupling_vs_kinematics_timing/code/arbitration_j_many.py", "03a_arbitration_sys_path.diff"),
                  ("03_j_coupling_vs_kinematics_timing/code/test_collective_j_timing.py", "03b_test_collective_sys_path.diff")]:
    d = subprocess.run(["diff", "-u", "--label", f"a/{rel}", "--label", f"b/{rel}", f"{ROOT}/{rel}", f"{P}/{rel}"],
                       capture_output=True, text=True).stdout
    open(f"patches/{name}", "w").write(d)
    print(name, len(d.splitlines()), "lines")

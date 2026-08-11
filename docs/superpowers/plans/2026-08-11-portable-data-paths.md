# Portable Data Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the selected personal cluster path defaults with portable repository-relative defaults while preserving their existing directory suffixes and interfaces.

**Architecture:** This is a literal configuration-default change across two tracked Python modules and one intentionally untracked CLI file. The command-line flags and `DatasetBuilder` constructor remain unchanged; only their default path strings change. Verification uses a source contract because the repository's tests are currently ignored and the change introduces no runtime logic.

**Tech Stack:** Python 3.10, `argparse`, `pathlib.Path`, PowerShell verification, Git.

## Global Constraints

- Preserve `/scratch/peirong/kxu56/atlas` as the relative suffix `atlas`.
- Preserve `/scratch/peirong/kxu56/USB/assets/uncond` as the relative suffix `USB/assets/uncond`.
- Preserve `/scratch/peirong/kxu56/USB/assets/uncond_gen` as the relative suffix `USB/assets/uncond_gen`.
- Do not make `--real-root` or `--gen-root` required.
- Do not change argument names, constructor parameters, eager path resolution, missing-directory behavior, ignored tests, or unrelated user changes.
- Keep `split_train_val.py` untracked unless the user separately requests a tracking change.

---

### Task 1: Replace personal path defaults

**Files:**
- Modify: `train.py:16`
- Modify: `seg_training/data.py:141-142`
- Modify: `split_train_val.py:12` (currently untracked)
- Test: inline PowerShell source contract; do not create or restore ignored test files

**Interfaces:**
- Consumes: `train.py` flag `--gen-root`, `split_train_val.py` flag `--real-root`, and `DatasetBuilder(real_root=..., gen_root=...)`.
- Produces: the same flags and constructor parameters with defaults `USB/assets/uncond`, `atlas`, and `USB/assets/uncond_gen`.

- [ ] **Step 1: Run the source contract before implementation and verify RED**

```powershell
$checks = @(
    @{ Path = 'train.py'; Expected = 'default="USB/assets/uncond"' },
    @{ Path = 'seg_training/data.py'; Expected = 'real_root: Path = Path("atlas")' },
    @{ Path = 'seg_training/data.py'; Expected = 'gen_root: Path = Path("USB/assets/uncond_gen")' },
    @{ Path = 'split_train_val.py'; Expected = 'default="atlas"' }
)
foreach ($check in $checks) {
    $text = Get-Content -LiteralPath $check.Path -Raw
    if (-not $text.Contains($check.Expected)) {
        throw "Missing expected portable default in $($check.Path): $($check.Expected)"
    }
}
$forbidden = Select-String -LiteralPath train.py,seg_training/data.py,split_train_val.py -SimpleMatch '/scratch/peirong/kxu56'
if ($forbidden) {
    throw 'Personal cluster prefix is still present.'
}
```

Expected: the command fails with `Missing expected portable default in train.py` because the personal defaults are still present.

- [ ] **Step 2: Apply the minimal replacements**

Change `train.py` to:

```python
parser.add_argument("--gen-root", default="USB/assets/uncond")
```

Change the `DatasetBuilder` defaults in `seg_training/data.py` to:

```python
real_root: Path = Path("atlas"),
gen_root: Path = Path("USB/assets/uncond_gen"),
```

Change `split_train_val.py` to:

```python
parser.add_argument("--real-root", default="atlas")
```

- [ ] **Step 3: Rerun the source contract and verify GREEN**

Run the complete PowerShell command from Step 1.

Expected: exit code `0` with no exception.

- [ ] **Step 4: Verify scope and inspect the resulting content**

```powershell
Select-String -LiteralPath train.py,seg_training/data.py,split_train_val.py -SimpleMatch '/scratch/peirong/kxu56'
git diff -- train.py seg_training/data.py
Select-String -LiteralPath split_train_val.py -Pattern 'real-root'
git status --short --untracked-files=all
```

Expected:

- The prefix scan prints no matches.
- The tracked diff contains only the three approved replacements in `train.py` and `seg_training/data.py`.
- The untracked `split_train_val.py` line shows `default="atlas"`.
- Existing unrelated status entries remain unchanged.

- [ ] **Step 5: Commit only tracked implementation files without changing split tracking**

```powershell
git add -- train.py seg_training/data.py
git commit --only -m "fix: use portable data path defaults" -- train.py seg_training/data.py
```

Expected: the commit contains only `train.py` and `seg_training/data.py`; `split_train_val.py` remains untracked and the user's pre-existing staged deletions remain staged.

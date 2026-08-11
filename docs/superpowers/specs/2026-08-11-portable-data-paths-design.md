# Portable Data Paths Design

## Goal

Remove the personal cluster prefix `/scratch/peirong/kxu56/` from the selected project files while preserving each existing repository-relative directory suffix and all current command-line behavior.

## Scope

Modify only these path defaults:

- `train.py`: `/scratch/peirong/kxu56/USB/assets/uncond` becomes `USB/assets/uncond`.
- `seg_training/data.py`: `/scratch/peirong/kxu56/atlas` becomes `atlas`.
- `seg_training/data.py`: `/scratch/peirong/kxu56/USB/assets/uncond_gen` becomes `USB/assets/uncond_gen`.
- `split_train_val.py`: `/scratch/peirong/kxu56/atlas` becomes `atlas`, even though this file is currently untracked.

Ignored test files and unrelated user changes are outside this change.

## Behavior

All affected values remain neutral relative-path defaults. No argument becomes required, no directory suffix changes, and no path is resolved eagerly. Existing callers may continue overriding the defaults through the same command-line arguments or constructor parameters.

## Error Handling

No new validation or error behavior is introduced. Missing data directories continue to be handled by the existing dataset discovery behavior.

## Verification

Use a test-first source check covering the four approved mappings, observe it fail against the current personal paths, then apply the minimal replacements and rerun it. Finally, scan `train.py`, `seg_training/data.py`, and `split_train_val.py` to confirm that `/scratch/peirong/kxu56` no longer appears, and inspect the Git diff to ensure no unrelated file content changed.

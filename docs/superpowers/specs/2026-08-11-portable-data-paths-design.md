# Portable Data Paths Design

## Goal

Remove the personal cluster prefix from the selected project files while preserving each existing repository-relative directory suffix and all current command-line behavior.

## Scope

Modify only these path defaults:

- `train.py`: the synthetic-data default becomes `USB/assets/uncond`.
- `seg_training/data.py`: the real-data default becomes `atlas`.
- `seg_training/data.py`: the generated-data default becomes `USB/assets/uncond_gen`.
- `split_train_val.py`: the real-data default becomes `atlas`, even though this file is currently untracked.

Ignored test files and unrelated user changes are outside this change.

## Behavior

All affected values remain neutral relative-path defaults. No argument becomes required, no directory suffix changes, and no path is resolved eagerly. Existing callers may continue overriding the defaults through the same command-line arguments or constructor parameters.

## Error Handling

No new validation or error behavior is introduced. Missing data directories continue to be handled by the existing dataset discovery behavior.

## Verification

Use test-first behavior checks covering the four approved defaults, observe them fail against the current personal paths, then apply the minimal replacements and rerun them. Finally, scan `train.py`, `seg_training/data.py`, and `split_train_val.py` to confirm that the personal prefix no longer appears, and inspect the Git diff to ensure no unrelated file content changed.

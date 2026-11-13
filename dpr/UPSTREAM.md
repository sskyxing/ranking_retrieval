# Upstream provenance

`official/` is an unmodified source snapshot exported from
[`facebookresearch/DPR`](https://github.com/facebookresearch/DPR) for the
EMNLP 2020 reproduction path.

- Commit: `d7ba973ab2d862d1b44c2402b59b1ca2a362ee96`
- Commit date: 2020-06-30
- Git tree: `37e64abb2e8cb9952532e37ca96845a8fb6a8eab`
- Upstream license: CC BY-NC 4.0; see [official/LICENSE](official/LICENSE)
- Export command: `git archive d7ba973ab2d862d1b44c2402b59b1ca2a362ee96`

The export was unpacked on Windows: 25 text files use CRLF instead of the
upstream blobs' LF endings. Replacing CRLF with LF reproduces every upstream
blob exactly; no source content was otherwise changed. The Git tree above is
therefore a normalized-content provenance identifier, not a byte-for-byte tree
hash of this Windows checkout.

This snapshot is deliberately older than upstream `main`. The latter contains
the March 2021 adversarial-hard-negative model and later Hydra/API changes, so
mixing its checkpoints, embeddings, commands, or reported metrics with the
2020 paper path would not be a faithful reproduction.

Files outside `official/` are local reproduction helpers and are not part of
the upstream snapshot.

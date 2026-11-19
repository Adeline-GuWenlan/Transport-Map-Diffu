# third_party (clones and downloads are gitignored; read-only, never imported or executed)

| Dir | Source | Commit / version | License | Used for |
|---|---|---|---|---|
| `ddm/` | https://github.com/edluyuan/ddm | `8a43f77d9bb528937c96a712d34abf5473af2c3f` (2025-09-22) | **none in repo** (no LICENSE file): read for the audit only, no code copied | audit (`docs/audit.md` §2) |
| `ffjord/` | https://github.com/rtqichen/ffjord (`--depth 1`) | `994864ad0517db3549717c25170f9b71e96788b1` | MIT | checkerboard generator, re-implemented in `src/ddm2d/data.py` with attribution |
| `arxiv_src/` | https://arxiv.org/e-print/2502.02483 and https://arxiv.org/pdf/2502.02483 | v3 (7 Jul 2025); tarball sha256 `f989b28cc0d66895d382190f215736776b1ad54905bbe15749ad55f7ba1a5f54`, PDF sha256 `1db1c19dfb6ad2709b1f59292c439452117c756da14df9fae577670ed3cd87eb` | arXiv (authors' copyright) | audit |
| `galashov_src/` | https://arxiv.org/e-print/2405.06780 | v1; tarball sha256 `d8cf1c9cc678bb1e1f5a42314b4738efcb1e70e42e565154b5845439a31d0abc` | arXiv | audit §4 |

Re-fetch: `git clone https://github.com/edluyuan/ddm.git ddm && git -C ddm checkout 8a43f77`;
`git clone https://github.com/rtqichen/ffjord.git ffjord && git -C ffjord checkout 994864a`; `curl -L -o src.tar.gz https://arxiv.org/e-print/<id>`.

Tool added outside the env: `pypdf 7.1.1` in `/scratch/wg2381/.cache-home/ddm2d_site` (pip `--target`), used only to extract text from the
paper PDF for equation numbers. Not needed by the package.

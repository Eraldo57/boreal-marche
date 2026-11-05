# Defence preparation verification — 2026-11-05

This records checks of the local working tree while aligning presentation preparation with the
original 31-page Capstone_Project1.pdf. It is not a CI run, clean-clone reproduction, deployment,
new training experiment or independent model-quality evaluation.

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| Non-integration suite | 416 passed; 7 integration tests deselected | [Test log](logs/defence_unit_tests_20261105.log) |
| Artifact integration suite | 7 passed; 416 other tests deselected | [Integration log](logs/defence_integration_tests_20261105.log) |
| Total across the two disjoint selections | **423 passed** | Same working tree; selections `not integration` and `integration` |
| Combined coverage, `src` and `api` | **90%**, 2,999 statements, 289 missed; passed 80% gate | [Coverage report](defence_coverage_20261105.txt) |
| Ruff lint | Passed | `ruff check src api frontend scripts tests` |
| Ruff formatting | Passed; 90 files already formatted | `ruff format --check src api frontend scripts tests` |
| Mypy | No issues in 33 source files | `mypy src api --ignore-missing-imports` |

The initial combined test run was interrupted after 295 passes because Hugging Face's HTTP model
check stalled in the restricted network environment. It was not a full passing run. The complete
verification above uses a separate non-integration run and a network-enabled integration run.
See [the interrupted attempt](logs/defence_tests_20261105.log). Its coverage data was not used.
The current selected coverage data was stored separately under `/tmp`; the report is retained here.
Existing dependency/deprecation warnings remain in the logs.

After updating the coverage-reference wording from the original PDF, the three
`tests/test_basket_conclusions.py` checks were rerun successfully. The model/rules were not
retrained, no threshold or registry alias was changed, and no cloud deployment was initiated.

## Repository/external status

- `origin` is configured for `Eraldo57/boreal-marche` on GitHub. Older “no remote” statements are stale.
- `gh repo view` and `gh run list` returned HTTP 401: Bad credentials. Therefore current repository
  visibility, pushed changes, workflow conclusions, branch protection and releases were not verified.
- `git tag --list` returned no local tags. This does not establish whether remote-only tags exist.
- The local working tree contains substantial pre-existing changes and the current presentation/UI
  additions. Local passing checks do not prove that the remote head has those changes or passes CI.
- Historical container sizes predate the latest UI/report additions. No fresh image build/size or
  different-machine reproduction was performed in this preparation task.

## Document checks

The timed walkthrough and cue card were aligned with the presentation/defence format in pp. 16–17,
the rubrics in pp. 21–22, the checklist on p. 28 and Appendix C on pp. 30–31. All question numbers
4–32 are represented in the updated preparation notes. Requirements gaps are mapped to their PDF
sections in [the completion checklist](../../docs/CAPSTONE_REQUIREMENTS_STATUS.md).

The original Appendix B (p. 29) does quote coverage above 40%, but does not identify catalogue versus
rule-firing coverage. The Streamlit note was corrected to state that ambiguity rather than say the
original rubric had not been read. No grade or business-effectiveness claim follows from this check.

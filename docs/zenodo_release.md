# Zenodo release procedure

1. Ensure CI passes on the intended release commit.
2. Confirm `pyproject.toml`, `CITATION.cff`, `.zenodo.json` and `CHANGELOG.md` agree on the release version and repository name.
3. Confirm no generated caches, private data, trained models or large outputs are tracked.
4. Create an annotated GitHub release tag, initially `v0.1.0`.
5. Archive that immutable GitHub release through Zenodo.
6. Add the Zenodo DOI badge and DOI citation to `README.md`/`CITATION.cff` in the next commit; use the concept DOI for the software generally and the version DOI when citing the exact archived release.
7. Record the BAMPS version/DOI in the companion analysis repository and manuscript Code Availability statement.

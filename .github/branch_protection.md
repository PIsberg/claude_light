# Branch Protection Settings

This document describes the required branch protection settings for the `main` branch.

## Required Settings

Enable the following branch protection rules in GitHub Settings → Branches → Add branch protection rule:

**Branch name pattern:** `main`

### Rule Configuration

- [x] **Require a pull request before merging**
  - [x] Require approvals: `1`
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners (if CODEOWNERS exists)

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - Required status checks:
    - `test (3.11)`
    - `dependency-review`
    - `sbom`
    - `CodeQL`

- [x] **Do not allow bypassing the above settings**
  - Applies to everyone including administrators

- [x] **Allow force pushes** → ❌ Disabled (no force pushes)

- [x] **Allow deletions** → ❌ Disabled (cannot delete branch)

- [x] **Allow trusted users to bypass** → ❌ Disabled (optional, based on team)

- [x] **Block creation of merge commits** → Optional (prefer squash/rebase)

## How to Configure

1. Go to **Settings** → **Branches** → **Branch protection rules**
2. Click **Add branch protection rule**
3. Enter `main` as the branch name pattern
4. Check all the boxes above
5. Click **Create**

## Why This Matters

Branch protection ensures:
- All code changes are reviewed by at least one person
- CI tests must pass before merging
- No one can force push or delete the main branch
- Consistent code quality and security standards

These settings are required for a high OpenSSF Scorecard score.

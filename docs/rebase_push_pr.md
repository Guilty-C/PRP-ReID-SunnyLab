# Resolving Non-Fast-Forward Pushes for `feat/exp003-benchmark`

This guide documents the exact steps required to re-synchronize the local
`feat/exp003-benchmark` branch with the remote repository before opening the
pull request for the augmentation work. The commands mirror the release
engineering checklist and are safe to run in Git Bash on Windows.

## 1. Fetch the latest remote commits

```bash
git fetch origin
git branch --set-upstream-to=origin/feat/exp003-benchmark feat/exp003-benchmark
git status
```

Use the following command to visualize which commits are only on your machine
or only on GitHub:

```bash
git log --oneline --graph --decorate --left-right --cherry-pick \
  origin/feat/exp003-benchmark...HEAD
```

## 2. Rebase on top of the remote branch

Keep the branch history linear by rebasing local commits on top of the remote
tracking branch:

```bash
git rebase origin/feat/exp003-benchmark
```

If conflicts appear, resolve them in your editor, stage the fixes, and then run
`git rebase --continue`. To bail out entirely, use `git rebase --abort`.

## 3. Push safely with `--force-with-lease`

After a successful rebase, update the remote branch while ensuring you do not
overwrite new commits pushed by collaborators:

```bash
git push --force-with-lease
```

If the remote has moved since the last fetch, Git will refuse the push, letting
you fetch and rebase again.

## 4. Open the pull request in the browser

Navigate to the comparison URL and verify the base and compare branches before
submitting the PR form with the prepared title and body:

```
https://github.com/Guilty-C/PRP-ReID-SunnyLab/compare/main...feat/exp003-benchmark
```

Add the labels `component:tuner`, `type:feature`, and `status:needs-review` if
they exist in the repository, and request reviewers as needed.

## 5. Troubleshooting

* **Credential prompts on HTTPS** – provide a GitHub Personal Access Token with
  the `repo` scope when Git asks for a password.
* **Remote commits keep reappearing** – another collaborator may be pushing to
  the same branch. Fetch again and rebase to incorporate their work.
* **CI failures after pushing** – confirm the smoke and unit tests still pass
  locally before retrying the pipeline.

Following this checklist ensures the branch matches the remote history and that
GitHub displays the augmentation changes exactly once in the pull request.

# GitHub Collaboration Guide — Sign Language Translation

**Repository:** https://github.com/amnisymimi/sign-language-translation

**Project:** Deep Learning — Sign Language Translation

This guide explains how each team member should set up GitHub, work on their own branch, upload files, and collaborate without accidentally overwriting other members' work.

## 1. Accept the GitHub Invitation

Before starting, make sure you have accepted the repository invitation.

1. Log in to https://github.com.
2. Check your GitHub notifications or email.
3. Find the invitation to `amnisymimi/sign-language-translation`.
4. Click **Accept invitation**.

After accepting, you should be able to access the repository link above.

## 2. Install Git and VS Code

Install the following software if you haven't already:

- Git: https://git-scm.com/downloads
- VS Code: https://code.visualstudio.com

Open VS Code, then select **Terminal → New Terminal**.

Check whether Git is installed:

```powershell
git --version
```

If Git is installed, the terminal will display its version number.

## 3. Configure Git (First-Time Setup)

Run the following commands using your own name and GitHub-associated email address:

```powershell
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

For example:

```powershell
git config --global user.name "teammate1"
git config --global user.email "your-email@example.com"
```

Verify the configuration:

```powershell
git config --global user.name
git config --global user.email
```

This identifies who authored each commit. It does not replace GitHub authentication.

## 4. Create a Local Project Folder

Choose where you want to save the project on your laptop.

For example, on Windows:

```powershell
mkdir "D:\DeepLearning"
cd "D:\DeepLearning"
```

If the folder already exists, simply run the `cd` command.

You can choose any location you prefer.

## 5. Clone the Shared GitHub Repository

Run:

```powershell
git clone https://github.com/amnisymimi/sign-language-translation.git
```

This downloads the repository to your laptop.

Then enter the repository:

```powershell
cd sign-language-translation
```

Open it in VS Code:

```powershell
code .
```

Your local folder should look like:

```text
DeepLearning/
└── sign-language-translation/
    ├── .gitignore
    └── README.md
```

Additional project folders will appear as the team adds files.

**Important:** Do not create a separate GitHub repository or run `git init`. Everyone should clone the same shared repository.

## 6. Create Your Own Development Branch

Each member must work on their own branch.

| Member | Branch |
|---|---|
| Person 1 | `teammate1-dev` |
| Person 2 | `teammate2-dev` |
| Person 3 | `syamimi-dev` |

Person 3's branch has already been created.

### Person 1

Run:

```powershell
git switch main
git pull origin main
git switch -c teammate1-dev
git push -u origin teammate1-dev
```

### Person 2

Run:

```powershell
git switch main
git pull origin main
git switch -c teammate2-dev
git push -u origin teammate2-dev
```

Each person should run only the commands for their own branch.

If the branch already exists on GitHub, use:

```powershell
git fetch origin
git switch --track origin/YOUR_BRANCH
```

Replace `YOUR_BRANCH` with your assigned branch name.

### Verify Your Branch

Run:

```powershell
git branch
```

For Person 1, the output should resemble:

```text
* teammate1-dev
  main
```

The asterisk (`*`) indicates your current branch.

## 7. Create Your Local Directories

All members should use the same repository structure.

Proposed structure:

```text
sign-language-translation/
├── preprocessing/
│   └── ...
├── features/
│   ├── manual/
│   └── nonmanual/
├── models/
│   ├── encoders/
│   ├── fusion/
│   └── decoder/
├── training/
├── evaluation/
├── configs/
├── docs/
├── README.md
└── .gitignore
```

### Creating Folders in VS Code

1. Open the repository folder in VS Code.
2. Right-click the project folder in Explorer.
3. Select **New Folder**.
4. Enter the folder name.
5. Right-click the new folder and select **New File**.

Alternatively, use PowerShell:

```powershell
New-Item -ItemType Directory -Force "models\fusion"
```

**Important:** Git does not track empty folders. Add a file such as `README.md` before committing a new folder.

Only create folders you actually need. Avoid creating duplicate folders for the same component.

## 8. Add or Modify Your Code

Create your Python files inside the appropriate directories.

For example:

**Person 1 — Dataset and Manual Features**

```text
preprocessing/
    prepare_dataset.py
    extract_hand_keypoints.py
```

**Person 2 — Facial Features and Fusion**

```text
features/nonmanual/
    extract_facial_features.py

models/fusion/
    temporal_alignment.py
    cross_attention.py
```

**Person 3 — Translation Model**

```text
models/decoder/
    translation_decoder.py

training/
    train_baseline.py
```

These filenames are suggestions and can be adjusted by the team.

After writing code, save the files in VS Code using `Ctrl + S`.

## 9. Upload Your Files to GitHub

Once you finish editing your code, open the VS Code terminal.

### Step 1 — Check Your Branch

```powershell
git branch
git status
```

Make sure you are on your own development branch.

### Step 2 — Stage Your Files

For example:

```powershell
git add models/fusion/
```

Or stage a specific file:

```powershell
git add models/fusion/cross_attention.py
```

Check which files are staged:

```powershell
git diff --cached --name-only
```

### Step 3 — Commit Changes

```powershell
git commit -m "Implement cross-attention module"
```

Use meaningful commit messages that describe what you changed.

### Step 4 — Push to GitHub

For Person 1:

```powershell
git push origin teammate1-dev
```

For Person 2:

```powershell
git push origin teammate2-dev
```

For Person 3:

```powershell
git push origin syamimi-dev
```

Your changes are now uploaded to your branch on GitHub.

## 10. Create a Pull Request

When your feature is ready to be integrated into the shared project:

1. Open the GitHub repository.
2. Go to **Pull requests**.
3. Click **New pull request**.
4. Set `base` to `main`.
5. Set `compare` to your development branch.
6. Add a title and description.
7. Click **Create pull request**.
8. Ask another teammate to review your code.
9. After review and testing, merge the pull request into `main`.

Do not merge incomplete or untested code into `main`.

## 11. Get the Latest Changes from Teammates

When someone merges changes into `main`, update your local repository.

First, save or commit your current work.

Then:

```powershell
git switch main
git pull origin main
```

To bring those changes into your own branch:

```powershell
git switch YOUR_BRANCH
git merge main
```

Replace `YOUR_BRANCH` with your branch name.

If Git reports a merge conflict, resolve it carefully in VS Code before completing the merge.

After merging, push your updated branch:

```powershell
git push origin YOUR_BRANCH
```

## 12. Daily GitHub Workflow

Every time you start working:

```powershell
cd "PATH_TO_YOUR_REPOSITORY"
git switch YOUR_BRANCH
git pull origin YOUR_BRANCH
```

Before editing, check your GitHub Issues to see which task you're responsible for.

After finishing a coding session:

```powershell
git status
git add PATH_TO_CHANGED_FILES
git commit -m "Describe your changes"
git push origin YOUR_BRANCH
```

Replace the placeholders with your actual paths and branch.

If there are no changes, you do not need to commit.

## 13. GitHub Project Task Management

We use GitHub Projects to track our five-week development plan.

Each member should:

1. Open the shared GitHub Project.
2. Find their assigned task for the current week.
3. Move it from **Todo** to **In Progress** when starting.
4. Update the issue checklist as tasks are completed.
5. Link relevant pull requests to the issue.
6. Move the task to **Done** only when its deliverable is completed and verified.

If a task is blocked, leave a comment explaining the problem so other teammates can help.

## 14. Important Rules

- Always check your branch before editing or pushing.
- Do not push directly to `main`.
- Do not use `git push --force` on shared branches.
- Do not upload large datasets, videos, or model checkpoints.
- Never upload passwords, access tokens, or AWS `.pem` keys.
- Use pull requests for integrating completed work.
- Agree on shared feature formats before implementing dependent components.
- Inform teammates when changing model interfaces or shared data formats.
- Review code before merging.

## 15. Quick Troubleshooting

**`Repository not found`**

Check that you accepted the invitation, copied the correct repository URL, and authenticated using the invited GitHub account.

**`Author identity unknown`**

Configure both your Git username and email:

```powershell
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

**`pathspec did not match any files`**

Check that the file or directory exists and that you are inside the correct repository.

**`Everything up-to-date`**

There are no new commits to push. Check `git status` to see whether you still need to stage and commit files.

**`Updates were rejected`**

Your local branch may be behind the remote branch. Fetch the latest changes and reconcile them before pushing again. Do not force-push.

---

## Final Setup Checklist

- [ ] Accepted GitHub repository invitation
- [ ] Installed Git and VS Code
- [ ] Configured Git username and email
- [ ] Cloned the shared repository
- [ ] Created personal development branch
- [ ] Pushed branch to GitHub
- [ ] Created necessary local directories
- [ ] Successfully committed and pushed a test file
- [ ] Confirmed access to the shared GitHub Project
- [ ] Understood the pull request workflow

Once all steps are complete, you're ready to collaborate!

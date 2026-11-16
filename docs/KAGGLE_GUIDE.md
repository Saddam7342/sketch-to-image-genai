# Running the experiments on Kaggle (free GPU), step by step

Kaggle gives every verified account about **30 GPU hours per week** for free. The full experiment run of this
project needs about **1 hour**. Kaggle occasionally renames buttons, so the labels below may differ slightly.

---

## Step 1: Create and verify your account (one time, ~5 min)

1. Go to <https://www.kaggle.com> and **Register** (Google sign-in is fine).
2. Click your profile picture (top right) → **Settings**.
3. Find **Phone verification** and verify your number.
   *Without this, Kaggle does not allow GPU or Internet access in notebooks.*

## Step 2: Make the code available to Kaggle

Pick **one** option.

**Option A: GitHub (recommended; you need the repo for the submission anyway)**
Push this project to a GitHub repository. The repository must be **public**, or Kaggle cannot clone it
without extra setup. Copy its URL, e.g. `https://github.com/<you>/sketch2image.git`.

**Option B: Kaggle Dataset (no GitHub needed)**
1. Zip the project folder (without `.venv` and `results`).
2. On Kaggle: **Create** (left sidebar, "+") → **New Dataset** → drag the zip in → give it a title →
   keep it **Private** → **Create**.

## Step 3: Import the notebook

1. On Kaggle: **Create** → **New Notebook**.
2. In the notebook menu: **File → Import Notebook** → upload `notebooks/kaggle_run_all.ipynb`
   from this project.

## Step 4: Turn on the GPU and the Internet

In the right-hand panel (**Session options** / **Settings**; open it with the `>|` icon if hidden):

| Setting | Value |
|---|---|
| **Accelerator** | **GPU T4 x2** (a single T4 is also fine; avoid P100) |
| **Internet** | **On** (needed to download the models from Hugging Face) |
| Language / Environment | leave the defaults |

If you chose **Option B**: in the same panel click **+ Add Input** → **Your Datasets** → select your dataset.

## Step 5: Point the notebook at your code

- **Option A:** in the cell "1. Get the code", `REPO_URL` already points to the project repository; change it only if you fork the project.
- **Option B:** nothing to change; the notebook finds the attached dataset automatically.

## Step 6: Quick test (interactive, ~5 min)

Run the first four cells one by one (click a cell, then **Shift + Enter**):

1. `nvidia-smi` should show a **Tesla T4**.
2. The code is cloned or copied.
3. Packages are installed (red pip "dependency conflict" warnings are normal on Kaggle and can be ignored).
4. The sanity check downloads the models (~3 GB, first time only) and shows **one robot image**.

If you see the robot, everything works. You can stop the interactive session now (power icon, top right),
which saves GPU quota.

## Step 7: Full run in the background (~60–75 min)

1. Top right: **Save Version**.
2. Choose **Save & Run All (Commit)** → **Save**.
3. Kaggle now runs the whole notebook on its own machine. **You can close the browser.**
4. Follow progress from your notebook page → **Versions** / the running version's log.

## Step 8: Download the results

1. Open the finished version of the notebook → **Output** tab.
2. Download **`results.zip`**.
3. Unzip it and copy the contained `results/` folder into this project folder (next to `src/`).

Then, locally:

```bash
python scripts/analyze.py   # regenerates plots/tables, already done on Kaggle too
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `AssertionError: No GPU!` | Accelerator is set to *None*: Step 4. |
| `Could not resolve host` / download errors | Internet is *Off*, or the phone is not verified: Steps 1 and 4. |
| `CUDA out of memory` in the SDXL cell | Add `--cpu-offload` to that cell's command and run it again. |
| Run stopped halfway (quota, timeout) | Simply run again: finished images are skipped automatically. |
| `git clone` asks for a password | The GitHub repo is private: make it public or use Option B. |
| Weekly GPU quota used up | It resets weekly; the quota counter is in your Kaggle settings/profile. |

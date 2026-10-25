# Deploy the Student Performance dashboard

This guide uses **Streamlit Community Cloud**. The prepared folder is `Streamlit_Deployment_Ready`; it contains the updated dashboard, both saved prediction models, the data files the app reads, pinned Python dependencies, and the theme file in the repository root. **The app has not been uploaded or deployed yet.**

## 1. Decide who may see the data

The dashboard displays individual student records, scores, and some background fields from the supplied CSV. The dataset's original source and synthetic status have not been verified. Start with a **private** GitHub repository and a private Streamlit app unless you have permission to publish these records. A private app can be shared with named viewers. See [Streamlit's sharing guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## 2. Upload the prepared files to GitHub

1. Extract `Streamlit_Deployment_Ready.zip`. **Upload the extracted contents, not the ZIP file itself.**
2. Sign in to GitHub and create a new repository, for example `student-performance-prediction-system`. Select **Private** initially. Leave the GitHub README, license, and `.gitignore` options unchecked because those files are already in the prepared folder.
3. In the new repository, choose **Add file → Upload files** (or use the upload link shown on an empty repository).
4. Open the extracted `Streamlit_Deployment_Ready` folder in Finder. Press `Command + Shift + .` if hidden files are not visible. Select **all contents inside the folder** and drag them into GitHub's upload page. GitHub's browser upload supports folders and up to 100 files at once; this package has fewer than 100 files, each under 25 MiB. [GitHub upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
5. Commit the upload. At the repository root, confirm that you can see `.streamlit/`, `module1/`, `module2/`, `module3/`, `module4/`, `module5/`, and `README.md`. Inside `module3/artifacts/`, confirm that **both** `score_model.joblib` and `score_model_with_mock_marks.joblib` are present.

If the folders appear inside an extra `Streamlit_Deployment_Ready/` directory on GitHub, the contents were uploaded one level too deep. Move them to the repository root before deploying. The app entrypoint must be `module4/app.py` relative to the repository root.

## 3. Create the Streamlit app

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with the GitHub account that owns the repository.
2. Choose **Create app → Yup, I have an app**.
3. Select your repository and its `main` branch.
4. Set the **Main file path** to `module4/app.py`.
5. Open **Advanced settings** and select **Python 3.12**. No secrets are needed for this project.
6. Click **Deploy** and wait for the build to finish. Streamlit will give you a `streamlit.app` link. [Official deployment steps](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

Streamlit will use `module4/requirements.txt` because it is beside the app file. The `.streamlit/config.toml` file at the repository root supplies the dashboard theme. [Streamlit file layout](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), [dependency rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)

## 4. Check the live result

- **Overview:** charts load, and the default model shows 80.6% level accuracy.
- **Predict Performance → No:** generates a result with the original 11-input model.
- **Predict Performance → Yes:** requires all four earlier mock marks and generates a result with the separate model.
- **Model Insights:** shows the optional model's 93.95% test accuracy and the assumption about the timing of marks.
- **Study plan:** recommendations appear after a prediction; a weak subject may appear when one mock mark is sufficiently lower than the other three.

These scores are results on the existing project test set, which has been used in earlier iterations. They are not a guarantee for new real students. The optional model assumes the four subject marks were available before the final exam, as you confirmed; that timing is not independently verified by the CSV.

## If deployment fails

Open **Manage app → Logs** on Streamlit Cloud and check the first error. Common causes for this project are an extra folder level in the GitHub upload, a missing `.joblib`/CSV/JSON file, an incorrect main file path, or a missed dependency file. Keep the whole prepared folder structure together; the app loads files from sibling modules.

The app remains local until you complete the GitHub upload and Streamlit deployment steps yourself.

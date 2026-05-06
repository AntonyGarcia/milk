# Data

Place downloaded datasets here. The repository includes folder structure, READMEs, and header-only metadata placeholders, but not third-party medical images.

The script resolves relative paths inside each dataset root. For example, `data/isic2019/metadata/isic2019_train.csv` with `image_path=images/ISIC_0000001.jpg` points to `data/isic2019/images/ISIC_0000001.jpg`.

Use `templates/csv/` when building metadata CSV files.

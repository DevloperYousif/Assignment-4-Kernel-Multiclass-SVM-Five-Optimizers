# Dataset - UCI Human Activity Recognition Using Smartphones

| Item | Value |
|---|---|
| Official page (UCI ML Repository, id 240) | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| Direct download (zip) | https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip |
| Kaggle mirror (CSV format) | https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones |
| Citation | D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013 |
| Licence | Listed as CC BY 4.0 on the UCI repository page; the original README requires citing the paper above and states that commercial use is prohibited |

## Dataset included in this repository
The dataset is stored here as a split archive (`UCI_HAR_Dataset.zip.part000` ... `part003`, GitHub web uploads are
limited in file size). Rebuild it once from the project root:
```bash
python data/rebuild_dataset.py      # creates data/UCI HAR Dataset/
```
(Windows alternative: `copy /b data\UCI_HAR_Dataset.zip.part000+data\UCI_HAR_Dataset.zip.part001+data\UCI_HAR_Dataset.zip.part002+data\UCI_HAR_Dataset.zip.part003 data\UCI_HAR_Dataset.zip` and unzip into `data/`.)

## Verified content (checked by the code in `src/dataset_loader.py`)

| Partition | Windows | Subjects | Features | Classes |
|---|---|---|---|---|
| official train | 7,352 | 21 | 561 | 6 |
| official test | 2,947 | 9 (different people) | 561 | 6 |

Classes (from `activity_labels.txt`): 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING.

SHA-256 of the files used for the reported results:
- `train/X_train.txt`: `4fcabaf84f390742b7d87722c75f0956665db9f44992e4b31fb3555cec21f15c`
- `test/X_test.txt`: `99035209add50d17650e847d8a4658d784a5707466ebb7c2deb97c28e5250a78`

(The UCI host was not reachable from the build sandbox, so the files were obtained from an unmodified mirror of the
original archive on Hugging Face (`udayl/UCI_HAR`); the row counts, subject lists, labels and README match the
official distribution.)

## Where to put the files

**Local / VS Code:** unzip so that this structure exists:
```
data/
└── UCI HAR Dataset/
    ├── activity_labels.txt
    ├── features.txt
    ├── train/  X_train.txt  y_train.txt  subject_train.txt
    └── test/   X_test.txt   y_test.txt   subject_test.txt
```
(`UCI HAR Dataset.zip` from the UCI zip is nested inside the downloaded zip - unzip both.)

**Kaggle:** click *Add Input* and attach `uciml/human-activity-recognition-with-smartphones`; the loader finds
`/kaggle/input/.../train.csv` and `test.csv` automatically (columns `subject` and `Activity` are recognised).

You can also point the loader at any folder with the environment variable `HAR_DATA_DIR`.

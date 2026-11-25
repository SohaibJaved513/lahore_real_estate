# Dataset

`raw/lahore_house_listings_zameen.csv` is the exact dataset supplied for this project.

The training workflow creates `processed/lahore_house_listings_clean.csv`. It normalizes prices and areas, validates counts, derives safe title flags, separates detailed location from broad society, removes likely duplicate listings and creates model-ready features.

Do not manually edit the processed file. Regenerate it with:

```bash
python run_pipeline.py
```

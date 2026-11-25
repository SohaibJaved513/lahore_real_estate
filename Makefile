install:
	python -m pip install -r requirements.txt

train:
	python run_pipeline.py

quick:
	python run_pipeline.py --quick

app:
	python app.py

test:
	pytest -q

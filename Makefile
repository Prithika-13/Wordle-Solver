.PHONY: test baselines train eval plots app

test:
	python -m pytest -q

baselines:
	python -m src.evaluate random frequency entropy entropy-cand

train:
	python scripts/train_rl.py --mode blind --seed 0 --iters 300
	python scripts/train_rl.py --mode assisted --seed 0 --iters 300

eval:
	python scripts/compare.py

plots:
	python scripts/plots.py

app:
	streamlit run app/streamlit_app.py

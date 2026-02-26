setup:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

run-naive: setup
	./venv/bin/python word2vec.py --mode naive
	
run-optimized: setup
	./venv/bin/python word2vec.py --mode optimized
	
run-compare: setup
	./venv/bin/python word2vec.py --mode compare
	
clean:
	rm -rf venv __pycache__
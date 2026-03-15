run: run-compare

venv:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

run-naive: venv
	./venv/bin/python word2vec.py --mode naive
	
run-optimized: venv
	./venv/bin/python word2vec.py --mode optimized
	
run-compare: venv
	./venv/bin/python word2vec.py --mode compare
	
clean:
	rm -rf venv __pycache__
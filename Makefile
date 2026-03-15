.PHONY: help install train api dashboard test clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train model"
	@echo "  make api        - Run API"
	@echo "  make dashboard  - Run Dashboard"
	@echo "  make test       - Run tests"

install:
	pip install -r requirements.txt

train:
	python src/model/train.py

api:
	python src/api/app.py

dashboard:
	streamlit run src/dashboard/dashboard.py

test:
	pytest tests/ -v

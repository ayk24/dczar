lint:
	poetry run isort .
	poetry run black .
	poetry run mypy src/dialogue
	poetry run mypy src/zar

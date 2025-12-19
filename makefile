
# -----------------------------------------------------

setup:
	sudo apt update --yes
	sudo apt upgrade --yes
	sudo apt-get remove swig
	sudo apt-get install swig3.0
	sudo ln -sf /usr/bin/swig3.0 /usr/bin/swig
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt install python3.11 python3.11-dev python3.11-venv python3-pip --yes
	python3.11 -m venv venv

install:
	pip3 install -U pip
	pip3 install build
	pip install poetry
	poetry install


# -----------------------------------------------------

autotune:
	python src/triton_gated_mlp/autotune.py

benchmark:
	python src/triton_gated_mlp/benchmark.py


# -----------------------------------------------------

test:
	poetry run pytest -m "not slow" -vs

style:
	poetry run black src/triton_gated_mlp
	


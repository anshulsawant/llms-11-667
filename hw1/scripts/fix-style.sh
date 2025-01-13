#!/bin/bash

ruff format src/ tests/ scripts/*.py
# ruff check --select I --fix src/ tests/
ruff check --fix  src/

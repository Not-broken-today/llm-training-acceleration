#!/bin/bash
echo "Creating virtual environment..."
python3.12 -m venv .venv_llm
echo "Activate virtual environment..."
source .venv_llm/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Installing project in development mode..."
pip install -e .
echo "Done!"
read -p "Press enter to exit..."
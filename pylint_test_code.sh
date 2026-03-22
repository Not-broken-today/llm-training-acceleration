#!/bin/bash
echo "Start test pylint code"
echo "Activate virtual environment..."
source .venv_llm/bin/activate
pylint src/
echo "Done!"
read -p "Press enter to exit..."
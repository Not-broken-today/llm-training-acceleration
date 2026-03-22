@echo off
echo Creating virtual enviroment...
python3.12 -m venv .venv_llm

echo Activate virtual enviroment...
call ".venv_llm\Scripts\activate"

echo Installing dependencies...
pip install -r requirements.txt

echo Installing project in development mode...
pip install -e .

echo Done!
pause
@echo off
echo Start test pylint code

echo Activate virtual enviroment...
call ".venv_llm\Scripts\activate"

call pylint src/

echo Done!
pause
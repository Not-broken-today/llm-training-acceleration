@echo off
cd ..
".venv_llm\Scripts\python.exe" -m src.main active_optimizer='adamw'
".venv_llm\Scripts\python.exe" -m src.main active_optimizer='muon'
".venv_llm\Scripts\python.exe" -m src.main active_optimizer='hybrid'
pause
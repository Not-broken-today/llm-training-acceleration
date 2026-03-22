#!/bin/bash
cd ..
.venv_llm/bin/python -m src.main active_optimizer='adamw'
.venv_llm/bin/python -m src.main active_optimizer='muon'
.venv_llm/bin/python -m src.main active_optimizer='hybrid'
read -p "Press enter to exit..."
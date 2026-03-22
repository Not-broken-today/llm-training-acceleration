#!/bin/bash
cd ..
.venv_llm/bin/python -m src.main active_optimizer='adamw'
read -p "Press enter to exit..."
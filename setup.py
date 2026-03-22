"""Setup script for LoRA_LLM_Huawei package."""
from setuptools import setup, find_packages

setup(
    name="lora_llm_huawei",
    version="0.0.1",
    description="LoRA Fine-tuning Pipeline for LLMs",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
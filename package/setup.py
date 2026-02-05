from setuptools import setup, find_packages

setup(
    name="eff-len",
    version="0.1.0",
    description="Effective sequence length (L_eff) and spectral selection for MSAs",
    author="Vaitea Opuu",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy"],
    scripts=["bin/leff"],
)

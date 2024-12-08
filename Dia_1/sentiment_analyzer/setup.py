from setuptools import setup,find_packages

setup(
    name='sentiment_analyzer',
    version='1.0.0',
    description='libreria para clasificar textos en positivos negativos y neutros',
    author='Gerardo Grande',
    author_email='elcoleccionista28gc@gmail.com',
    packages=find_packages(),
    include_package_data=True,  # Incluir archivos no Python
    package_data={
        "sentiment_analyzer": ["../sentimet_model/*"],  # Archivos del modelo
    },
    install_requires=[
        "transformers>=4.0.0",
        "numpy>=1.18.0",
        "torch"
    ],
)
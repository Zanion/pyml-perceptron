from setuptools import setup, find_packages

setup(
    name="pyml-perceptron",
    version='0.0.1',
    py_modules=["pyml_perceptron"],
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['test']),
    include_package_data=True,
    install_requires=[
        'click',
        'numpy'
    ],
    entry_points='''
        [console_scripts]
        pyml-perceptron=pyml_perceptron.cli:cli
    ''',
)

import sys
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyml_perceptron.defaults import IRIS_DATASET
from pyml_perceptron.app import App
from pyml_perceptron.core.perceptron import Perceptron


OPT = {}


@click.command()
def cli():
	# Set all provided arguments to configuration
	app = App(**OPT)
	return app.run()


@click.command()
def train():
    df = pd.read_csv(IRIS_DATASET, header=None)

    # Select setosa and versicolor
    y = df.iloc[[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # Extract sepal length and petal length


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
    df = pd.read_csv(IRIS_DATASET, headers=None)


import sys
import click
from pyml-perceptron.app import App


OPT = {}


@click.command()
def cli():
	# Set all provided arguments to configuration
	app = App(**OPT)
	return app.run()


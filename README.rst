
.. raw:: html

	<p align="center">
		<img src="./logo/logo_lce.svg" width="35%">	
	</p>
	
	<div align="center">
		<a href="https://circleci.com/gh/LocalCascadeEnsemble/LCE/tree/main">
			<img src="https://circleci.com/gh/LocalCascadeEnsemble/LCE/tree/main.svg?style=shield">
		</a>
		<a href="https://lce.readthedocs.io/en/latest/?badge=latest">
			<img src="https://readthedocs.org/projects/lce/badge/?version=latest">
		</a>
		<a href="https://pypi.python.org/pypi/lcensemble/">		
			<img src="https://badge.fury.io/py/lcensemble.svg">
		</a>		
		<a href="https://pypi.python.org/pypi/lcensemble/">		
			<img src="https://img.shields.io/pypi/pyversions/lcensemble.svg">
		</a>
		<a href="https://github.com/psf/black">	
			<img src="https://img.shields.io/badge/code%20style-black-000000.svg">
		</a>
		<a href="https://pypi.python.org/pypi/lcensemble/">		
			<img src="https://img.shields.io/github/license/LocalCascadeEnsemble/LCE.svg">
		</a>
	</div>
   
**Local Cascade Ensemble (LCE)** is a *high-performing*, *scalable* and *user-friendly* machine learning method for the general tasks of **classification** and **regression**.
In particular, LCE:
 
- Enhances the prediction performance of Random Forest and XGBoost by combining their strengths and adopting a complementary diversification approach
- Supports parallel processing to ensure scalability
- Handles missing data by design
- Adopts scikit-learn API for the ease of use
- Adheres to scikit-learn conventions to allow interaction with scikit-learn pipelines and model selection tools
- Is released in open source and commercially usable - Apache 2.0 license

A tutorial introducing LCE and illustrative code examples has been published in `Towards Data Science <https://towardsdatascience.com/random-forest-or-xgboost-it-is-time-to-explore-lce-2fed913eafb8?source=friends_link&sk=8cba14ad36f7662d07e842d03944a316>`_.

Getting Started
===============

This section presents a quick start tutorial showing snippets for you to try out LCE.

Installation
------------

You can install LCE from `PyPI <https://pypi.org/project/lcensemble/>`_ with ``pip``::

	pip install lcensemble
	
Or ``conda``::

	conda install -c conda-forge lcensemble
	
	
First Example on Iris Dataset
-----------------------------

LCEClassifier prediction on an Iris test set:

.. code-block:: python

	from lce import LCEClassifier
	from sklearn.datasets import load_iris
	from sklearn.metrics import classification_report
	from sklearn.model_selection import train_test_split


	# Load data and generate a train/test split
	data = load_iris()
	X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

	# Train LCEClassifier with default parameters
	clf = LCEClassifier(n_jobs=-1, random_state=0)
	clf.fit(X_train, y_train)

	# Make prediction and generate classification report
	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred))


Documentation
=============

LCE documentation, including API documentation and general examples, can be found `here <https://lce.readthedocs.io/en/latest/>`_.


Reference
=========

The full information about LCE design and evaluation can be found in the associated `journal paper <https://hal.inria.fr/hal-03599214/document>`_:

.. [1] Fauvel, K., E. Fromont, V. Masson, P. Faverdin and A. Termier. "XEM: An Explainable-by-Design Ensemble Method for Multivariate Time Series Classification", Data Mining and Knowledge Discovery, 36(3):917–957, 2022
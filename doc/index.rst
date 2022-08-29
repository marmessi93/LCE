LCE Documentation
====================

.. raw:: html
	
	<div align="center">
		<a href="https://circleci.com/gh/LocalCascadeEnsemble/LCE/tree/main">
			<img src="https://circleci.com/gh/LocalCascadeEnsemble/LCE/tree/main.svg?style=shield">
		</a>
		<a href="https://codecov.io/gh/LocalCascadeEnsemble/LCE">
			<img src="https://codecov.io/gh/LocalCascadeEnsemble/LCE/branch/main/graph/badge.svg?token=VTA64P4GTF">
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
   
| **Local Cascade Ensemble (LCE)** is a *high-performing*, *scalable* and *user-friendly* machine learning method for the general tasks of **Classification** and **Regression**.
| In particular, LCE:
 
- Enhances the prediction performance of Random Forest and XGBoost by combining their strengths and adopting a complementary diversification approach
- Supports parallel processing to ensure scalability
- Handles missing data by design
- Adopts scikit-learn API for the ease of use
- Adheres to scikit-learn conventions to allow interaction with scikit-learn pipelines and model selection tools
- Is released in open source and commercially usable - Apache 2.0 license

An article introducing LCE and illustrative code examples has been published in `Towards Data Science <https://towardsdatascience.com/random-forest-or-xgboost-it-is-time-to-explore-lce-2fed913eafb8?source=friends_link&sk=8cba14ad36f7662d07e842d03944a316>`_.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorial

   tutorial

.. toctree::
   :maxdepth: 2
   :hidden:	
   :caption: Documentation

   api
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   reference
   

`Tutorial <tutorial.html>`_
-------------------------------------
 
This section is a tutorial presenting LCE algorithm, its installation procedure and some general code examples.

As a first example, after having installed LCE package using ``pip`` or ``conda`` and loaded a train and test set, the ``classification_report`` from scikit-learn can be obtained with 4 lines of code:

.. code-block:: python

	# Train LCEClassifier with default parameters
	clf = LCEClassifier(n_jobs=-1, random_state=0)
	clf.fit(X_train, y_train)

	# Make prediction and generate classification report
	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred))
 

`API Documentation <api.html>`_
-------------------------------

This section contains the API documentation of LCE; it details ``LCEClassifier`` and ``LCERegressor`` attributes, methods and parameters. 


`Reference <reference.html>`_
--------------------------------------

The full information about LCE design and evaluation can be found in the associated `journal paper <https://hal.inria.fr/hal-03599214/document>`_:

.. [1] Fauvel, K., E. Fromont, V. Masson, P. Faverdin and A. Termier. XEM: An Explainable-by-Design Ensemble Method for Multivariate Time Series Classification. Data Mining and Knowledge Discovery, 36(3):917–957, 2022
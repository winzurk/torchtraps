=====================================
Torch Traps :leopard: :camera_flash:
=====================================


.. image:: https://img.shields.io/pypi/v/torchtraps.svg
        :target: https://pypi.python.org/pypi/torchtraps

.. image:: https://img.shields.io/travis/winzurk/torchtraps.svg
        :target: https://travis-ci.com/winzurk/torchtraps

.. image:: https://readthedocs.org/projects/torchtraps/badge/?version=latest
        :target: https://torchtraps.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Python package for lighting :zap: fast wildlife camera trap image annotation based on PyTorch. :fire:


* MIT license
* Documentation: https://torchtraps.readthedocs.io.

Install
--------
.. code-block:: bash

    $ pip install torchtraps

Run on Folder of Images
-------------------------------

.. code-block:: python

    import torchtraps as tt
    from torchtraps.coco_camera_traps_loader import images_from_dir

    images = images_from_dir('/PathToCameraTrapImageFolder/')
    model = tt.get_model(task='classification', level='species')
    tt.predict_to_csv(model, images, 'SavePredictions.csv')

Features
--------

* Simple module for computer vision on camera trap images.
* Based on PyTorch



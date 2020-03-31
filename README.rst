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

Fast Inference on Folder of Images
-------------------------------------------------

.. code-block:: python

    import torchtraps

    torchtraps.lightning.kachow('path/to/image/folder')


.. csv-table:: Example Output
    :header: "image", "prediction", "confidence"

        "image1.jpg", "jaguar", 0.99
        "image2.jpg", "empty", 0.98
        "image3.jpg", "agouti", 0.91
        "image4.jpg", "empty", 0.95
        "image5.jpg", "ocelot", 0.87



Features
--------

* Module for fast computer vision on camera trap images.
* Based on PyTorch



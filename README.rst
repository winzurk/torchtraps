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


Torch Traps is python package for *lighting* :zap: *fast* wildlife camera trap image annotation based on PyTorch. :fire:

* Documentation: https://torchtraps.readthedocs.io.
* GitHub: https://github.com/winzurk/torchtraps
* PyPI: https://pypi.python.org/pypi/torchtraps
* MIT License

.. torchtraps/sample_images/NJP-2.JPG

.. image:: https://github.com/winzurk/torchtraps/blob/master/torchtraps/sample_images/NJP-2.JPG
        :target: https://github.com/winzurk/torchtraps/blob/master/torchtraps/sample_images/NJP-2.JPG
        :width: 300

Photo Credit: Northern Jaguar Project

Over the past several decades, biologists all over the world have widely adopted camera traps as a standard tool for
monitoring biodiversity, resulting in backlogs often on the order of millions of images waiting to be manually reviewed
by humans to assess wildlife population densities. The application of modern computer vision and deep learning methods
to accelerate the processing of wildlife camera trap data has the potential to alleviate existing bottlenecks for large
scale biodiversity monitoring, thus dramatically increasing the speed at which researchers can obtain data-driven
insights about ecosystems, and ultimately leading to more efficient resource allocation and better informed policy
making by NGOs and government agencies.

Torch Traps aims to provide a simple tool (as little as 1 line of code) to bring state-of-the-art computer vision models
into the hands of biologists to accelerate the speed at which they can review camera trap imagery.


Install
--------
.. code-block:: bash

    $ pip install torchtraps

Fast Inference on Folder of Images
-------------------------------------------------

Classify an entire folder of camera trap images in one line of code by simply passing the relative path to the folder
containing images. Outputs are automatically saved to a csv file which can be further processed opened in an application
like Excel.

.. code-block:: python

    import torchtraps.lightning import kachow

    kachow('path/to/image/folder')


.. csv-table:: Example Output File
    :header: "image", "prediction", "confidence"

        "image1.jpg", "jaguar", 0.99
        "image2.jpg", "empty", 0.98
        "image3.jpg", "agouti", 0.91
        "image4.jpg", "empty", 0.95
        "image5.jpg", "ocelot", 0.87



Features
--------

* Module for fast computer vision on camera trap images.
* Train and fine-tune classification models on your own dataset.
* Based on PyTorch
* MIT license


Complete Installation Tutorial from Scratch
----------------------------------------------
This is a full tutorial on how to install and get up and running with Torch Traps. Zero programming knowledge is
assumed in the attempt to make Torch Traps as accessible as possible. If you do run into any problems, please email
me at zwinzurk@asu.edu

* Step 1: Install Anaconda

    Go to https://www.anaconda.com/distribution/

    Download Anaconda Python 3.7 version for the operating system you are using (Windows, MacOS, or Linux).

    Click on 64-Bit Graphical Installer (442 MB) to download the version with a Graphical User Interface.

    .. image:: tutorial/AnacondaDownload.jpg
        :width: 300

    Why do I need Anaconda?

        Torch Traps is a module written in `Python <http://www.python.org/>`_ (a programming language), so we first need to have Python installed
        on our computer. There are several ways to install python, but Anaconda allows us to install Python and it comes
        pre-installed with many of the common modules used for Data Science, and optionally comes with a GUI which can
        be used to open notebooks.

    After download is complete, double-click to install and follow installation instructions.

    .. image:: tutorial/InstallAnaconda.jpg
        :width: 300


* Step 2: Open Anaconda Navigator

    After installing Anaconda, open the Anaconda Navigator application on your computer.

    .. image:: tutorial/OpenNavigator.jpg
        :width: 300

* Step 3: Launch Jupyter Lab

    We will then launch a Jupyter Lab. Your web browser will open but the Jupyter server is running locally as you can
    see the url should be http://localhost:8889/lab

    .. image:: tutorial/LaunchJupyter.jpg
        :width: 300

* Step 4: Navigate to Working Folder on Left

    By clicking on the folder icon in the upper-left corner we can navigate the file system.

    Navigate to the directory on your computer where your camera trap image folder is located.

* Step 5: Open Python3 Notebook

    Now that we are working in the right directory, we can launch a new Python notebook. This will create a new file in
    our working directory called Untitled.ipynb. We can right-click on the file name to re-name it.

    .. image:: tutorial/CreateNotebook.jpg
        :width: 300

* Step 6: Install Torch Traps

    Jupyter notebooks allow us to run python code one 'cell' at a time. So the first thing we need to do is install
    torch traps, if we have not before. Copy the code below into the first cell, and then run the cell by either
    clicking the play button or hitting SHIFT+ENTER at the same.

    .. code-block:: bash

        !pip install torch traps

    .. image:: tutorial/InstallTorchTraps.jpg
        :width: 300

 * Step 7: Run Torch Traps on Folder of Images

    Now that the Torch Traps is installed, you can copy the code below into a new code cell.

    Change the 'path/to/image/folder' to the name of your folder containing camera trap images (ex. 'camera_trap_images')

    Run the cell. (SHIFT + ENTER)

    Note: If running for the first time, an internet connection will be required to download the model file.

    When complete an output.csv file will appear in the directory you are working in. You can double-click csv files to
    view in Jupyter Lab or open with another application like Excel.

    .. code-block:: python

        import torchtraps.lightning import kachow
        kachow('path/to/image/folder')


    .. image:: tutorial/RunTorchTraps.jpg
        :width: 300

* Step 8: Open CSV File To See Classification Results

.. Future: Step 9: View Images of Particular Class

















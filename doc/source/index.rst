.. xiuminglib documentation master file, created by
   sphinx-quickstart on Mon Mar 25 15:24:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xiuminglib's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

xiuminglib includes daily classes and functions that are useful for my computer vision/graphics research.
Noteworthily, it contains many useful functions for 3D modeling and rendering with Blender.

The source code is available in `the repo <https://github.com/xiumingzhang/xiuminglib>`_. For issues or
questions, please open an issue there.


.. include:: modules.rst


Installation
============

Simply clone the repo and add it to your ``PYTHONPATH``.

.. code-block:: bash

    cd <your_local_dir>
    git clone https://github.com/xiumingzhang/xiuminglib.git
    export PYTHONPATH=<your_local_dirdir>/xiuminglib/:$PYTHONPATH

Dependencies
------------

Depending on what functions you want to use, you may need to install:

    OpenCV 3.3.0
        If you use conda, it's as easy as ``conda install -c conda-forge opencv=3.3.0``.

    Matplotlib 2.0.2
        Some functions are known to be buggy with 3.0.0. If you use conda, do
        ``conda install -c conda-forge matplotlib=2.0.2``.

    Blender
        Note this is different from installing Blender as an application, which has Python bundled.
        Rather, this is installing Blender as a Python module: you've succeeded if you can
        ``import bpy`` in the Python you use. I did this "the hard way":
        `building it from source <https://blender.stackexchange.com/a/117213/30822>`_,
        but with hindsight, `a one-liner <https://anaconda.org/kitsune.one/python-blender>`_
        *may* work just as well.

        If ``import bpy`` throws ``Segmentation fault``, try again with Python 3.6.3.

    Trimesh
        If you use conda, it's as simple as ``conda install -c conda-forge trimesh``.

The library uses "on-demand" imports whenever possible, so that it won't fail on imports that you don't need. 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

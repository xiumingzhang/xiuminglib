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

Simply clone the repo and add it to your ``PYTHONPATH``:

.. code-block:: bash

    cd <your_local_dir>
    git clone https://github.com/xiumingzhang/xiuminglib.git
    export PYTHONPATH="<your_local_dir>/xiuminglib/":"$PYTHONPATH"

The library is being developed and tested with Python 3.6.3.

Dependencies
------------

Besides super standard packages (like NumPy), you need:

    SciPy
        If you use conda, it's as easy as ``conda install scipy``.

    OpenCV 3.3.0
        Do ``conda install -c conda-forge opencv=3.3.0``. If any ``lib*.so*`` is missing at runtime,
        the easiest fix is to ``conda install`` the missing library to the same environment, maybe
        followed by some symlinking (like linking ``libjasper.so`` to ``libjasper.so.1``) inside
        ``<anaconda_dir>/envs/<env_name>/lib``. This is cleaner and easier than ``apt-get``, which
        may break other things and usually requires ``sudo``.

    Matplotlib 2.0.2
        Some functions are known to be buggy with 3.0.0. If you use conda, do
        ``conda install -c conda-forge matplotlib=2.0.2``.

Depending on what functions you want to use, you may also need to install:

    Blender 2.79
        Note this is different from installing Blender as an application, which has Python bundled.
        Rather, this is installing Blender as a Python module: you've succeeded if you find ``bpy.so``
        in the build's bin folder and can ``import bpy`` in your Python (not the Blender-bundled
        Python) after you add it to your ``PYTHONPATH``.

        I did this "the hard way": first building all dependencies from source, and then
        `building Blender from source <https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu>`_
        with ``-DWITH_PYTHON_MODULE=ON`` for CMake, primarily because I wanted to build to an NFS
        location so that a cluster of machines on the NFS can all use the build.

        If you only need Blender on a local machine, for which you can ``sudo``, then dependency
        installations are almost automatic -- just run ``install_deps.sh``, although when I did this,
        I had to ``skip-osl`` to complete the run, for some reason I didn't take time to find out.

        Blender 2.80 made some API changes that are incompatible with this library, so please make sure
        you check out
        `the correct tag <https://git.blender.org/gitweb/gitweb.cgi/blender.git/tag/refs/tags/v2.79b>`_
        with ``git checkout``, followed by ``git submodule update`` to ensure the submodules are of
        the correct versions.

        If ``import bpy`` throws ``Segmentation fault``, try again with Python 3.6.3.

    Trimesh
        See `their installation guide <https://github.com/mikedh/trimesh/blob/master/docs/install.rst>`_.

The library uses "on-demand" imports whenever possible, so that it won't fail on imports that you don't need.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

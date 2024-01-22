.. _install:

Installation
============

Dependencies
------------

LBParticles depends on ``matplotlib``, ``numpy``, ``scipy``, ``corner``, ``pandas``, and ``tqdm``. These will be
installed automatically when you install LBParticles through `pip <http://www.pip-installer.org/>`_, but they can
also be installed using the `requirements.txt` or `environment.yml` files at the root of the repository after cloning.

Using pip
---------

The easiest way to install the most recent stable version of ``LBParticles`` is
with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    python -m pip install LBParticles


From source
-----------

Alternatively, you can get the source by downloading a `tarball
<https://github.com/LBParticles/LBParticles/tarball/master>`_ or cloning `the git
repository <https://github.com/LBParticles/LBParticles>`_:

.. code-block:: bash

    git clone https://github.com/LBParticles/LBParticles.git

Once you've downloaded the source, you can navigate into the root source
directory and run:

.. code-block:: bash

    python -m pip install .


Running Tests
-------------

If you installed from source, you can run the unit tests. From the root of the
source directory, run:

.. code-block:: bash

    pytest
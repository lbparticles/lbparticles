.. _install:

Installation
============

Dependencies
------------

LBParticles depends on ``numpy`` and ``scipy``. 

These will be installed automatically when you install LBParticles through `pip <http://www.pip-installer.org/>`_.

Using pip
---------

The easiest way to install the most recent stable version of ``lbparticles`` is
with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    python -m pip install LBParticles


From source
-----------

Alternatively, you can get the source by downloading a `tarball
<https://github.com/lbparticles/lbparticles/tarball/master>`_ or cloning `the git
repository <https://github.com/lbparticles/lbparticles>`_:

.. code-block:: bash

    git clone https://github.com/lbparticles/lbparticles.git

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
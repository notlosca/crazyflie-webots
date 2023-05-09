Welcome to AI.CS
================

AI.CS is the coordinates transformation package in Python. It offers functionality for converting data between geometrical coordinates (cartesian, spherical and cylindrical) as well as between geocentric and heliocentric coordinate systems typically used in spacecraft measurements. The package currently also supports rotations of data by means of `rotation matrices <https://en.wikipedia.org/wiki/Rotation_matrix>`_. Transformations between spacecraft coordinate systems are implemented as a Python binding to the `CXFORM <https://spdf.sci.gsfc.nasa.gov/pub/software/old/selected_software_from_nssdc/coordinate_transform/>`_ library.

The full documentation is available at `aics.rtfd.io <http://aics.rtfd.io>`_.

Getting started
===============

This tutorial will guide you through basic usage of AI.CS.

Installation
------------

AI.CS is developed for Python 3, so make sure that you have a working isntallation of it. The package is distributed together with C portion of CXFORM library, which is compiled automatically during installation. Thus make sure that you have a functioning compiler in your system, for instance, gcc.

Assuming the above requirements are satisfied install the package with Python package manager::

    $ pip install ai.cs

Geometrical coordinates
-----------------------

AI.CS ships with functions for conversion between cartesian and spherical coordinates and between cartesian and cylindrical coordinates:

.. code-block:: python

    import numpy as np
    from ai import cs

    # cartesian to spherical
    r, theta, phi = cs.cart2sp(x=1, y=1, z=1)

    # spherical to cartesian
    x, y, z = cs.sp2cart(r=1, theta=np.pi/4, phi=np.pi/4)

    # cartesian to cylindrical
    r, phi, z = cs.cart2cyl(x=1, y=1, z=1)

    # cylindrical to cartesian
    x, y, z = cs.cyl2cart(r=1, phi=np.pi/2, z=1)

Most of the functions support both scalars and numpy arrays as input:

.. code-block:: python

    import numpy as np
    from ai import cs

    # converting spherical spiral from spherical to cartesian coordinates
    x, y, z = cs.sp2cart(
        r=np.ones(100),
        theta=np.linspace(-np.pi/2, np.pi/2, 100),
        phi=np.linspace(0, np.pi*6, 100)
    )

Spacecraft coordinates
----------------------

AI.CS provides Python bindings to CXFORM library for conversion between various geocentric and heliocentric cartesian coordinate systems. For example, the code below performs transformation of data from GSE to HEEQ coordinate system:

.. code-block:: python

    from datetime import datetime
    from astropy import units as u
    from ai import cs

    # converting (0.5, 0.5, 0.5) AU location from GSE to HEEQ at current time
    x, y, z = cs.cxform(
        'GSE',
        'HEEQ',
        datetime.now(),
        x=u.au.to(u.m, 0.5),
        y=u.au.to(u.m, 0.5),
        z=u.au.to(u.m, 0.5)
    )

Both scalars and numpy arrays are supported as input:

.. code-block:: python

    from datetime import datetime, timedelta
    from astropy import units as u
    from ai import cs

    # converting circular orbit at 1 AU from cylindrical to cartesian coordinates
    r = np.ones(365)*u.au.to(u.m, 1)
    phi = np.linspace(0, np.pi*2, 365)
    z = np.zeros(365)
    x_HEE, y_HEE, z_HEE = cs.cyl2cart(r, phi, z)

    # converting HEE to HEEQ
    x_HEEQ, y_HEEQ, z_HEEQ = cs.cxform(
        'HEE',
        'HEEQ',
        [datetime(2016, 1, 1)+timedelta(days=d) for d in range(365)],
        x=x_HEE,
        y=y_HEE,
        z=z_HEE
    )

Geometrical transformations
---------------------------

Currently AI.CS offers only one type of geometrical transformations - rotations. Rotation is executed by means of 3D transformation matrices for right-handed rotations around X, Y and Z axes: 

.. code-block:: python

    import numpy as np
    from ai import cs

    # get (3x3) rotation matrix for rotation by pi/4 around X axis 
    Tx = cs.mx_rot_x(gamma=np.pi/4)
    # get (3x3) rotation matrix for rotation by -pi/4 around Y axis 
    Ty = cs.mx_rot_y(theta=-np.pi/4)
    # get (3x3) rotation matrix for rotation by pi/2 around Z axis 
    Tz = cs.mx_rot_z(phi=np.pi/2)

Is is also possible to construct rotation matrices for compound rotations in one shot: 


.. code-block:: python

    import numpy as np
    from ai import cs

    # get matrix for right-handed rotation around X, Y and Z axes (exactly in this order)
    T = cs.mx_rot(theta=np.pi/4, phi=np.pi/4, gamma=np.pi/4)

    # get matrix for right-handed rotation around Z, Y and X axes (exactly in this order)
    T_reverse = cs.mx_rot_reverse(theta-np.pi/4, phi=-np.pi/4, gamma=-np.pi/4)
    # T_reverse effectively reverses the transformation described by T in this case

Rotation matrices can be applied to data in cartesian coordinates in the following way:

.. code-block:: python

    import numpy as np
    from ai import cs

    # a cube with the side length 2
    x = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    z = np.array([1, -1, 1, -1, 1, 1, -1, -1])

    # rotate cube by pi/4 around each axis
    T = cs.mx_rot(theta=np.pi/4, phi=np.pi/4, gamma=np.pi/4)
    x, y, z = cs.mx_apply(T, x, y, z)



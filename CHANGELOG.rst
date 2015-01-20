0.2.2 (2015-01-20)
------------------
* fixed mesh path
* use a pimpl to not let the user decide between GLUT or OSMesa
* remove legacy CMake
* Contributors: Vincent Rabaud, nlyubova

0.2.1 (2015-01-18)
------------------
* use GLUT by default
* clean extensions
* compile on Indigo
* fixed rotation matrix,
  fixed up vector
  additional option to render depth only
* fixed object orientation,
  return distance to an object is added
* build_depend on assimp-dev instead of assimp
* Fix Assimp detection
* Contributors: Scott K Logan, Vincent Rabaud, nlyubova

0.2.0 (2014-01-14  20:23:06 +0100)
----------------------------------
- get a 2d (planar) and a 3d (mesh) renderer
- drop Fuerte support

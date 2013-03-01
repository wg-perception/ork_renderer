.. _renderer:

Random view Generator
#####################

This package contains a small library to generate random views around an object. The input is given through a mesh and
views are generated according to certain boundaries (angles, scale, in-plane rotation). The outputs are
`OpenCV <http://opencv.org/>`_ ``cv::Mat`` for depth, RGB and mask.

Two versions exist: one based on `GLUT <http://www.opengl.org/resources/libraries/glut/>`_ and one on
`OSMesa <http://www.mesa3d.org/osmesa.html>`_. The first ones requires you have an OpenGL context and therefore a
window manager. The second is pure software and is therefore the one to use if you want to run the code on an
infrastructure without GUIS's (e.g. on a cluster). It is slower though.

By default, the GLUT one is compiled as OSMesa 9.0 seems to have a few problems these days.

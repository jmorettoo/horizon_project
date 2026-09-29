Black hole simulation using python and GLSL for the shaders
I took inspiration in youtuber Kavan010 who made a similar project in C++, I thought it was very cool so I decided to recreate it in python

Controls
  Left mouse drag : orbit the camera around the black hole
  Scroll wheel    : zoom in / out
  G               : turn gravity between the orbiting bodies on / off
  Right mouse     : hold to turn gravity on (like the original)
  Esc             : quit

It will need the following libraries: glfw PyOpenGL numpy PyGLM
Needs an OpenGL 4.3+ GPU (compute shaders). macOS does NOT support this.

Keep geodesic.comp in the same folder as this file.

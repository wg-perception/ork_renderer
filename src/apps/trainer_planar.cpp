//
// Copyright (c) 2012, Willow Garage, Inc.
// Copyright (c), assimp OpenGL sample
// Copyright (c) 2013, Aldebaran Robotics
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Willow Garage, Inc. nor the names of its
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#include <iostream>
#include <stdlib.h>

#include <boost/format.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <object_recognition_renderer/utils.h>
#include <object_recognition_renderer/renderer2d.h>

int main(int argc, char **argv) {
  // Define the display
  size_t width = 640, height = 480;

  // the model name can be specified on the command line.
  std::string file_name(argv[1]), file_ext = file_name.substr(file_name.size() - 3, file_name.npos);

  Renderer2d render(file_name, 0.2);
  double focal_length_x = 525, focal_length_y = 525;
  render.set_parameters(width, height, focal_length_x, focal_length_y);

  // Loop over a few views in front of the pattern
  float y = 0.6, z = 1;
  cv::Vec2f up(z, -y);
  up = up / norm(up);
  render.lookAt(0., y, z, 0, up(0), up(1));
  cv::Mat img, depth, mask;
  render.render(img, depth, mask);
  cv::imshow("img", img);
  cv::imshow("depth", depth);
  cv::imshow("mask", mask);
  cv::waitKey(0);

  return 0;
}

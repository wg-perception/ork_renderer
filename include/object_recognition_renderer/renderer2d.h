/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Vincent Rabaud
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef ORK_RENDERER_RENDERER2D_H_
#define ORK_RENDERER_RENDERER2D_H_

#include <string>

#include <opencv2/core/core.hpp>

#include "renderer.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that renders a planar scene under different view points
 * This is an equivalent to the Render class butfor planar patterns
 */
class Renderer2d : public Renderer
{
public:
  /**
   * @param file_path the path of the image file
   * @param scale the size of a pixel in real life (in meters)
   */
  Renderer2d(const std::string & file_path, float scale);

  void
  set_parameters(size_t width, size_t height, double focal_length_x, double focal_length_y, double near, double far);

  /** Similar to the gluLookAt function
   * @param x the x position of the eye pointt
   * @param y the y position of the eye point
   * @param z the z position of the eye point
   * @param upx the x direction of the up vector
   * @param upy the y direction of the up vector
   * @param upz the z direction of the up vector
   */
  void
  lookAt(GLdouble x, GLdouble y, GLdouble z, GLdouble upx, GLdouble upy, GLdouble upz);

  /** Renders the content of the current OpenGL buffers to images
   * @param image_out the RGB image
   * @param depth_out the depth image
   * @param mask_out the mask image
   */
  void
  render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out) const;

protected:
  /** Path of the mesh */
  std::string mesh_path_;

  unsigned int width_, height_;
  double focal_length_x_, focal_length_y_, near_, far_;
  float angle_;

  cv::Mat img_;
};

#endif /* ORK_RENDERER_RENDERER2D_H_ */

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
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

#ifndef ORK_RENDERER_RENDERER3D_H_
#define ORK_RENDERER_RENDERER3D_H_

#include <string>

#include <opencv2/core/core.hpp>

#include <GL/gl.h>

#include "renderer.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Model;
class aiLogStream;

class Renderer3dImpl;

/** Class that displays a scene in a Frame Buffer Object
 * Inspired by http://www.songho.ca/opengl/gl_fbo.html
 */
class Renderer3d : public Renderer
{
public:
  /**
   * @param file_path the path of the mesh file
   */
  Renderer3d(const std::string & file_path);

  virtual
  ~Renderer3d();

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
   * @param rect_out the bounding box of the rendered image
   */
  void
  render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out) const;

  /** Renders the depth image from the current OpenGL buffers
   * @param depth_out the depth image
   * @param mask_out the mask image
   * @param rect_out the bounding box of the rendered image
   */
  void
  renderDepthOnly(cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out) const;

  /** Renders the RGB image from the current OpenGL buffers
   * @param image_out the RGB image
   * @param rect_out the bounding box of the rendered image
   */
  void
  renderImageOnly(cv::Mat &image_out, const cv::Rect &rect_out) const;

protected:
  double focal_length_x_, focal_length_y_, near_, far_;
  float angle_;

  Model* model_;
  GLuint scene_list_;

  /** stream for storing the logs from Assimp */
  aiLogStream* ai_stream_;

  /** Private implementation of the renderer (GLUT or OSMesa) */
  Renderer3dImpl* renderer_;
};

#endif /* ORK_RENDERER_RENDERER3D_H_ */

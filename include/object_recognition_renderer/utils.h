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

#ifndef ORK_RENDERER_UTILS_H
#define ORK_RENDERER_UTILS_H

#include <string>

#include <opencv2/core/core.hpp>

class Renderer;

/** Function that normalizes a vector
 * @param x the x component of the vector
 * @param y the y component of the vector
 * @param z the z component of the vector
 */
template<typename T>
void normalize_vector(T & x, T&y, T&z)
{
  T norm = std::sqrt(x * x + y * y + z * z);
  x /= norm;
  y /= norm;
  z /= norm;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** Class that enables iterating the viewpoint over a sphere.
 * This function is used to generate templates in LINE-MOD
 */
class RendererIterator
{
public:
  /**
   * @param file_path the path of the mesh to render
   */
  RendererIterator(Renderer * renderer, size_t n_points);

  /** Iterate to get to a different view
   * We don't implement the postfix operator on purpose
   * @return an incremented version of itself
   */
  RendererIterator &
  operator++();

  /**
   * @return true if we are done with all the views, false otherwise
   */
  bool isDone() const
  {
    return (index_ >= n_points_);
  }

  /** Renders the content of the current OpenGL buffers to images
   * @param image_out the RGB image
   * @param depth_out the depth image
   * @param mask_out the mask image
   * @param rect_out the bounding box of the rendered image
   */
  void
  render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out);

  /** Renders the content of the current OpenGL buffers to images
   * @param image_out the RGB image
   * @param depth_out the depth image
   * @param mask_out the mask image
   * @param rect_out the bounding box of the rendered image
   * @param t the translation vector
   * @param up the up vector of the view point
   */
  void
  render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out, const cv::Vec3d &t, const cv::Vec3d &up);

  /** Renders the depth image from the current OpenGL buffers
   * @param depth_out the depth image
   * @param mask_out the mask image
   * @param rect_out the bounding box of the rendered image
   * @param t the translation vector
   * @param up the up vector of the view point
   */
  void
  renderDepthOnly(cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out, const cv::Vec3d &t, const cv::Vec3d &up);

  /** Renders the RGB image from the current OpenGL buffers
   * @param image_out the RGB image
   * @param rect_out the bounding box of the rendered image
   * @param t the translation vector
   * @param up the up vector of the view point
   */
  void
  renderImageOnly(cv::Mat &image_out, const cv::Rect &rect_out, const cv::Vec3d &t, const cv::Vec3d &up);

  /**
   * @return the rotation of the camera with respect to the current view point
   */
  cv::Matx33d
  R() const;

  /**
   * @return the rotation of the object with respect to the current view point
   */
  cv::Matx33d
  R_obj() const;

  /**
   * @return the distance from the current camera position to the object origin
   */
  float
  D_obj() const;

  /**
   * @return the translation of the camera with respect to the current view point
   */
  cv::Vec3d
  T() const;

  /**
   * @return the total number of templates that will be computed
   */
  size_t
  n_templates() const;

  /** The number of points on the sphere */
  size_t n_points_;
  /** The index of the view point we are at now */
  size_t index_;
  /** The renderer object containing the scene and that will render images */
  Renderer* renderer_;
  /** Values for the angle sampling in degrees */
  int angle_min_, angle_max_, angle_step_, angle_;
  /** Values for the scale sampling */
  float radius_min_, radius_max_, radius_step_, radius_;

private:
  /**
   * @param T the translation vector
   * @param up the up vector of the view point
   */
  void
  view_params(cv::Vec3d &T, cv::Vec3d &up) const;
};

#endif /* ORK_RENDERER_UTILS_H */

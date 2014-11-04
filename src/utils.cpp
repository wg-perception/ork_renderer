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

#include <object_recognition_renderer/renderer.h>
#include <object_recognition_renderer/utils.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RendererIterator::RendererIterator(Renderer *renderer, size_t n_points)
    :
      n_points_(n_points),
      index_(0),
      renderer_(renderer),
      angle_min_(-80),
      angle_max_(80),
      angle_step_(40),
      angle_(angle_min_),
      radius_min_(0.4),
      radius_max_(0.8),
      radius_step_(0.2),
      radius_(radius_min_)
{
}

RendererIterator &
RendererIterator::operator++()
{
  angle_ += angle_step_;
  if (angle_ > angle_max_)
  {
    angle_ = angle_min_;
    radius_ += radius_step_;
    if (radius_ > radius_max_)
    {
      radius_ = radius_min_;
      ++index_;
    }
  }

  return *this;
}

/**
 * @param image_out the RGB image
 * @param depth_out the depth image
 * @param mask_out the mask image
 * @param rect_out the bounding box of the rendered image
 */
void
RendererIterator::render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out)
{
  if (isDone())
    return;

  cv::Vec3d t, up;
  view_params(t, up);

  renderer_->lookAt(t(0), t(1), t(2), up(0), up(1), up(2));
  //renderer_->render(image_out, depth_out, mask_out, rect_out);
  renderer_->renderDepthOnly(depth_out, mask_out, rect_out);
  renderer_->renderImageOnly(image_out, rect_out);
}

/**
 * @param image_out the RGB image
 * @param depth_out the depth image
 * @param mask_out the mask image
 * @param rect_out the bounding box of the rendered image
 * @param t the translation vector
 * @param up the up vector of the view point
 */
void
RendererIterator::render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out, const cv::Vec3d &t, const cv::Vec3d &up)
{
  renderer_->lookAt(t(0), t(1), t(2), up(0), up(1), up(2));
  renderer_->renderDepthOnly(depth_out, mask_out, rect_out);
  renderer_->renderImageOnly(image_out, rect_out);
}

/**
 * @param depth_out the depth image
 * @param mask_out the mask image
 * @param rect_out the bounding box of the rendered image
 * @param t the translation vector
 * @param up the up vector of the view point
 */
void
RendererIterator::renderDepthOnly(cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out, const cv::Vec3d &t, const cv::Vec3d &up)
{
  renderer_->lookAt(t(0), t(1), t(2), up(0), up(1), up(2));
  renderer_->renderDepthOnly(depth_out, mask_out, rect_out);
}

/**
 * @param image_out
 * @param rect_out
 * @param t the translation vector
 * @param up the up vector of the view point
 */
void
RendererIterator::renderImageOnly(cv::Mat &image_out, const cv::Rect &rect_out, const cv::Vec3d &t, const cv::Vec3d &up)
{
  renderer_->lookAt(t(0), t(1), t(2), up(0), up(1), up(2));
  renderer_->renderImageOnly(image_out, rect_out);
}


/**
 * @return the rotation of the camera with respect to the current view point
 */
cv::Matx33d
RendererIterator::R() const
{
  cv::Vec3d t, up;
  view_params(t, up);
  normalize_vector(t(0),t(1),t(2));

  // compute the left vector
  cv::Vec3d y;
  y = up.cross(t);
  normalize_vector(y(0),y(1),y(2));

  // re-compute the orthonormal up vector
  up = t.cross(y);
  normalize_vector(up(0), up(1), up(2));

  cv::Mat R_full = (cv::Mat_<double>(3, 3) <<
                    t(0), t(1), t(2),
                    y(0), y(1), y(2),
                    up(0), up(1), up(2));
  cv::Matx33d R = R_full;
  R = R.t();

  return R;
}

/**
 * @return the rotation of the object with respect to the current view point
 */
cv::Matx33d
RendererIterator::R_obj() const
{
  cv::Vec3d t, up;
  view_params(t, up);
  normalize_vector(t(0),t(1),t(2));

  // compute the left vector
  cv::Vec3d y;
  y = up.cross(t);
  normalize_vector(y(0),y(1),y(2));

  // re-compute the orthonormal up vector
  up = t.cross(y);
  normalize_vector(up(0), up(1), up(2));

  cv::Mat R_full = (cv::Mat_<double>(3, 3) <<
                    -y(0), -y(1), -y(2),
                    -up(0), -up(1), -up(2),
                    t(0), t(1), t(2)
                    );

  cv::Matx33d R = R_full;
  R = R.t();

  return R.inv();
}

/**
 * @return the distance from the current camera position to the object origin
 */
float
RendererIterator::D_obj() const
{
  return radius_;
}

/**
 * @return the translation of the camera with respect to the current view point
 */
cv::Vec3d
RendererIterator::T() const
{
  cv::Vec3d t, _up;
  view_params(t, _up);

  return -t;
}

/**
 * @return the total number of templates that will be computed
 */
size_t
RendererIterator::n_templates() const
{
  return ((angle_max_ - angle_min_) / angle_step_ + 1) * n_points_ * ((radius_max_ - radius_min_) / radius_step_ + 1);
}

/**
 * @param T the translation vector
 * @param up the up vector of the view point
 */

void
RendererIterator::view_params(cv::Vec3d &T, cv::Vec3d &up) const
{
  float angle_rad = angle_ * CV_PI / 180.;

  // from http://www.xsi-blog.com/archives/115
  // compute the Point(x, y ,z) on the sphere based on index_ and radius_ using Golden Spiral technique
  static float inc = CV_PI * (3 - sqrt(5));
  static float off = 2.0f / float(n_points_);
  float y = index_ * off - 1.0f + (off / 2.0f);
  float r = sqrt(1.0f - y * y);
  float phi = index_ * inc;
  float x = std::cos(phi) * r;
  float z = std::sin(phi) * r;

  float lat = std::acos(z), lon;
  if ((fabs(std::sin(lat)) < 1e-5) || (fabs(y / std::sin(lat)) > 1))
    lon = 0;
  else
    lon = std::asin(y / std::sin(lat));

  x *= radius_; // * cos(lon) * sin(lat);
  y *= radius_; //float y = radius * sin(lon) * sin(lat);
  z *= radius_; //float z = radius * cos(lat);

  T = cv::Vec3d(x, y, z);

  // Figure out the up vector
  float x_up = radius_ * std::cos(lon) * std::sin(lat - 1e-5) - x;
  float y_up = radius_ * std::sin(lon) * std::sin(lat - 1e-5) - y;
  float z_up = radius_ * std::cos(lat - 1e-5) - z;
  normalize_vector(x_up, y_up, z_up);

  // Figure out the third vector of the basis
  float x_right = -y_up * z + z_up * y;
  float y_right = x_up * z - z_up * x;
  float z_right = -x_up * y + y_up * x;
  normalize_vector(x_right, y_right, z_right);

  // Rotate the up vector in that basis
  float x_new_up = x_up * std::cos(angle_rad) + x_right * std::sin(angle_rad);
  float y_new_up = y_up * std::cos(angle_rad) + y_right * std::sin(angle_rad);
  float z_new_up = z_up * std::cos(angle_rad) + z_right * std::sin(angle_rad);
  up = cv::Vec3d(x_new_up, y_new_up, z_new_up);

  // compute the left vector
  cv::Vec3d l;
  l = up.cross(T);  // cross product
  normalize_vector(l(0),l(1),l(2));

  up = T.cross(l);  // cross product
  normalize_vector(up(0), up(1), up(2));
}

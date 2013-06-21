/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Aldebaran Robotics
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

#include <iostream>

#include <object_recognition_renderer/renderer2d.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

Renderer2d::Renderer2d(const std::string & file_path, float physical_width) :
    mesh_path_(file_path), width_(640), height_(480), focal_length_x_(0), focal_length_y_(0), physical_width_(
        physical_width) {
  img_ori_ = cv::imread(file_path);
  if (img_ori_.channels() == 4) {
    // Get the alpha channel as the mask
    std::vector<cv::Mat> channels;
    cv::split(img_ori_, channels);
    channels[3].copyTo(mask_ori_);
    channels.resize(3);
    cv::merge(channels, img_ori_);
  } else
    mask_ori_ = cv::Mat_<uchar>::zeros(img_ori_.size());
}

Renderer2d::~Renderer2d() {
}

void Renderer2d::set_parameters(size_t width, size_t height, double focal_length_x, double focal_length_y) {
  width_ = width;
  height_ = height;

  focal_length_x_ = focal_length_x;
  focal_length_y_ = focal_length_y;

  K_ = cv::Matx33f(focal_length_x / 2, 0, width_ / 2, 0, focal_length_y / 2, height_ / 2, 0, 0, 1);
}

void Renderer2d::lookAt(GLdouble x, GLdouble y, GLdouble z, GLdouble upx, GLdouble upy, GLdouble upz) {
  cv::Matx33f R;
  cv::Vec3f T;

  // Update R_ and T_
  T_ = cv::Vec3f(x,y,z);
  cv::Vec3f up(upx, upy, upz);
  cv::Vec3f vec = cv::Vec3f(0, -1, 0).cross(up);
  float norm = cv::norm(vec);
  vec *= std::asin(norm) / norm;
  cv::Rodrigues(vec, R_);
}

void Renderer2d::render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out) const {
  // Figure out the transform from an original image pixel to a projected pixel
  // original frame: 0 is the top left corner of the pattern, X goes right, Y down, Z away from the camera
  // projected frame: 0 is the center of the projection image, X goes right, Y up, Z towards the camera

  // Scale the image properly
  float s = physical_width_ / img_ori_.cols;
  cv::Matx33f T_img = cv::Matx33f(s, 0, 0, 0, s, 0, 0, 0, 1);

  // Flip axes
  T_img = cv::Matx33f(1, 0, 0, 0, -1, 0, 0, 0, -1) * T_img;

  // Apply the camera transform
  cv::Matx34f P = K_
      * cv::Matx34f(R_(0, 0), R_(0, 1), R_(0, 2), T_(0), R_(1, 0), R_(1, 1), R_(1, 2), T_(1), R_(2, 0), R_(2, 1),
          R_(2, 2), T_(2));
  // Define the perspective transform to apply to the image (z=0 so we can ignore the 3rd column of P
  T_img = cv::Matx33f(P(0, 0), P(0, 1), P(0, 3), P(1, 0), P(1, 1), P(1, 3), P(2, 0), P(2, 1), P(2, 3)) * T_img;
  cv::Matx33f T_img_inv = T_img.inv();

  // Define the image corners
  std::vector<cv::Vec2f> corners(4);
  corners[0] = cv::Vec2f(0, 0);
  corners[1] = cv::Vec2f(physical_width_, 0);
  corners[2] = cv::Vec2f(physical_width_, (img_ori_.rows * physical_width_) / img_ori_.cols);
  corners[3] = cv::Vec2f(0, corners[2][1]);

  // Project the image corners
  float x_min = std::numeric_limits<float>::max(), y_min = x_min;

  std::vector<cv::Vec2f> corners_dst(4);
  for (size_t i = 0; i < corners.size(); ++i) {
    cv::Vec3f res = T_img * cv::Vec3f(corners[i][0], corners[i][1], 0);
    x_min = std::min(res[0] / res[2], x_min);
    y_min = std::min(res[1] / res[2], y_min);
  }

  // Warp the mask
  cv::Size final_size(width_, height_);
  cv::Mat_<uchar> mask;
  cv::warpPerspective(mask_ori_, mask, T_img, final_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

  // Warp the image/depth
  cv::Mat_<cv::Vec3b> image(final_size);
  cv::Mat_<unsigned short> depth(final_size);
  cv::Mat_<uchar>::iterator mask_it = mask.begin(), mask_end = mask.end();

  unsigned int i_min = width_, i_max = 0, j_min = height_, j_max = 0;
  for (unsigned int j = 0; j < height_; ++j)
    for (unsigned int i = 0; i < width_; ++i, ++mask_it) {
      if (!*mask_it)
        continue;
      // Figure out the coordinates of the original point
      cv::Vec3f point_ori = T_img_inv * cv::Vec3f(i, j, 1);

      int j_ori = point_ori[1] / point_ori[2], i_ori = point_ori[0] / point_ori[2];
      image(j, i) = img_ori_(j_ori, i_ori);

      // Figure out the 3d position of the point
      cv::Vec3f pos = R_ * cv::Vec3f(i_ori, j_ori, 1) + T_;
      // Do not forget to re-scale in millimeters
      depth(j, i) = pos[2]*1000;

      // Figure the inclusive bounding box of the mask, just for performance reasons for later
      if (j > j_max)
        j_max = j;
      else if (j < j_min)
        j_min = j;
      if (i > i_max)
        i_max = i;
      else if (i < i_min)
        i_min = i;
    }

  // Crop the images, just so that they are smaller to write/read
  if (i_min > 0)
    --i_min;
  if (i_max < width_ - 1)
    ++i_max;
  if (j_min > 0)
    --j_min;
  if (j_max < height_ - 1)
    ++j_max;
  cv::Rect rect(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);

  depth(rect).copyTo(depth_out);
  image(rect).copyTo(image_out);
  mask(rect).copyTo(mask_out);
}

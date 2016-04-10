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

#include <object_recognition_renderer/renderer2d.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <limits>

Renderer2d::Renderer2d(const std::string & file_path, float physical_width) :
    mesh_path_(file_path), width_(640), height_(480), focal_length_x_(0), focal_length_y_(0), physical_width_(
        physical_width) {
  cv::Mat img_ori = cv::imread(file_path, CV_LOAD_IMAGE_UNCHANGED);
  if (img_ori.channels() == 4) {
    // Get the alpha channel as the mask
    std::vector<cv::Mat> channels;
    cv::split(img_ori, channels);
    channels[3].copyTo(mask_ori_);
    channels.resize(3);
    cv::merge(channels, img_ori_);
  } else {
    img_ori_ = img_ori;
    mask_ori_.create(img_ori_.size());
    mask_ori_.setTo(255);
  }
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

void Renderer2d::lookAt(double x, double y, double z, double upx, double upy, double upz) {
  cv::Matx33f R;
  cv::Vec3f T;

  // Update R_ and T_
  T_ = cv::Vec3f(-x, -y, -z);
  cv::Vec3f f = T_/norm(T_);
  cv::Vec3f up(upx, upy, upz);
  up = up / norm(up);

  cv::Vec3f s = f.cross(up);
  cv::Vec3f u = s.cross(f);
  R_ = cv::Matx33f(s[0], s[1], s[2], u[0], u[1], u[2], -f[0], -f[1], -f[2]);
  R_ = R_.t();
  T_ = -R_*T_;
}

void Renderer2d::render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect) const {
  // Figure out the transform from an original image pixel to a projected pixel
  // original frame: 0 is the top left corner of the pattern, X goes right, Y down, Z away from the camera
  // projected frame: 0 is the center of the projection image, X goes right, Y up, Z towards the camera

  // Scale the image properly
  float s = physical_width_ / img_ori_.cols;
  cv::Matx44f T_img_physical = cv::Matx44f(s,0,0,0, 0,s,0,0, 0,0,s,0, 0,0,0,1);

  // Flip axes and center at 0,0,0
  float physical_height = img_ori_.rows * s;
  T_img_physical = cv::Matx44f(1,0,0,-float(physical_width_) / 2, 0,1,0,float(physical_height) / 2, 0,0,1,0, 0,0,0,1) *
      cv::Matx44f(1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1) * T_img_physical;

  // Define the perspective transform to apply to the image (z=0 so we can ignore the 3rd column of P)
  cv::Matx34f P_noK = cv::Matx34f(R_(0, 0), R_(0, 1), R_(0, 2), T_(0), R_(1, 0), R_(1, 1), R_(1, 2), T_(1), R_(2, 0),
      R_(2, 1), R_(2, 2), T_(2)) * T_img_physical;
  cv::Matx33f T_to3d = cv::Matx33f(P_noK(0, 0), P_noK(0, 1), P_noK(0, 3), P_noK(1, 0), P_noK(1, 1), P_noK(1, 3),
      P_noK(2, 0), P_noK(2, 1), P_noK(2, 3));

  // Apply the camera transform
  cv::Matx33f T_img = K_ * T_to3d;

  // And readapt to an OpenCV image
  T_img = cv::Matx33f(1, 0, 0, 0, -1, 0, 0, 0, 1) * T_img;

  // Define the image corners
  std::vector<cv::Vec2f> corners(4);
  corners[0] = cv::Vec2f(0, 0);
  corners[1] = cv::Vec2f(img_ori_.cols, 0);
  corners[2] = cv::Vec2f(img_ori_.cols, img_ori_.rows);
  corners[3] = cv::Vec2f(0, corners[2][1]);

  // Project the image corners
  float x_min = std::numeric_limits<float>::max(), y_min = x_min;

  std::vector<cv::Vec2f> corners_dst(4);
  for (size_t i = 0; i < corners.size(); ++i) {
    cv::Vec3f res = T_img * cv::Vec3f(corners[i][0], corners[i][1], 1);
    x_min = std::min(res[0] / res[2], x_min);
    y_min = std::min(res[1] / res[2], y_min);
  }

  // Warp the mask
  cv::Size final_size(width_, height_);
  cv::Mat_<uchar> mask;
  T_img = cv::Matx33f(1, 0, -x_min, 0, 1, -y_min, 0, 0, 1) * T_img;

  // Compute the inverse
  cv::Matx33f T_img_inv = T_img.inv();

  cv::warpPerspective(mask_ori_, mask, T_img, final_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

  // Warp the image/depth
  cv::Mat_<cv::Vec3b> image = cv::Mat_<cv::Vec3b>::zeros(final_size);
  cv::Mat_<unsigned short> depth = cv::Mat_<unsigned short>::zeros(final_size);
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
      cv::Vec3f pos = T_to3d * cv::Vec3f(i_ori, j_ori, 1);
      // Do not forget to re-scale in millimeters
      depth(j, i) = -pos[2]*1000;

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
  rect = cv::Rect(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);

  depth(rect).copyTo(depth_out);
  image(rect).copyTo(image_out);
  mask(rect).copyTo(mask_out);
}

void Renderer2d::renderDepthOnly(cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect) const {
  // Figure out the transform from an original image pixel to a projected pixel
  // original frame: 0 is the top left corner of the pattern, X goes right, Y down, Z away from the camera
  // projected frame: 0 is the center of the projection image, X goes right, Y up, Z towards the camera

  // Scale the image properly
  float s = physical_width_ / img_ori_.cols;
  cv::Matx44f T_img_physical = cv::Matx44f(s,0,0,0, 0,s,0,0, 0,0,s,0, 0,0,0,1);

  // Flip axes and center at 0,0,0
  float physical_height = img_ori_.rows * s;
  T_img_physical = cv::Matx44f(1,0,0,-float(physical_width_) / 2, 0,1,0,float(physical_height) / 2, 0,0,1,0, 0,0,0,1) *
      cv::Matx44f(1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1) * T_img_physical;

  // Define the perspective transform to apply to the image (z=0 so we can ignore the 3rd column of P)
  cv::Matx34f P_noK = cv::Matx34f(R_(0, 0), R_(0, 1), R_(0, 2), T_(0), R_(1, 0), R_(1, 1), R_(1, 2), T_(1), R_(2, 0),
      R_(2, 1), R_(2, 2), T_(2)) * T_img_physical;
  cv::Matx33f T_to3d = cv::Matx33f(P_noK(0, 0), P_noK(0, 1), P_noK(0, 3), P_noK(1, 0), P_noK(1, 1), P_noK(1, 3),
      P_noK(2, 0), P_noK(2, 1), P_noK(2, 3));

  // Apply the camera transform
  cv::Matx33f T_img = K_ * T_to3d;

  // And readapt to an OpenCV image
  T_img = cv::Matx33f(1, 0, 0, 0, -1, 0, 0, 0, 1) * T_img;

  // Define the image corners
  std::vector<cv::Vec2f> corners(4);
  corners[0] = cv::Vec2f(0, 0);
  corners[1] = cv::Vec2f(img_ori_.cols, 0);
  corners[2] = cv::Vec2f(img_ori_.cols, img_ori_.rows);
  corners[3] = cv::Vec2f(0, corners[2][1]);

  // Project the image corners
  float x_min = std::numeric_limits<float>::max(), y_min = x_min;

  std::vector<cv::Vec2f> corners_dst(4);
  for (size_t i = 0; i < corners.size(); ++i) {
    cv::Vec3f res = T_img * cv::Vec3f(corners[i][0], corners[i][1], 1);
    x_min = std::min(res[0] / res[2], x_min);
    y_min = std::min(res[1] / res[2], y_min);
  }

  // Warp the mask
  cv::Size final_size(width_, height_);
  cv::Mat_<uchar> mask;
  T_img = cv::Matx33f(1, 0, -x_min, 0, 1, -y_min, 0, 0, 1) * T_img;

  // Compute the inverse
  cv::Matx33f T_img_inv = T_img.inv();

  cv::warpPerspective(mask_ori_, mask, T_img, final_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

  // Warp the image/depth
  cv::Mat_<unsigned short> depth = cv::Mat_<unsigned short>::zeros(final_size);
  cv::Mat_<uchar>::iterator mask_it = mask.begin(), mask_end = mask.end();

  unsigned int i_min = width_, i_max = 0, j_min = height_, j_max = 0;
  for (unsigned int j = 0; j < height_; ++j)
    for (unsigned int i = 0; i < width_; ++i, ++mask_it) {
      if (!*mask_it)
        continue;
      // Figure out the coordinates of the original point
      cv::Vec3f point_ori = T_img_inv * cv::Vec3f(i, j, 1);

      int j_ori = point_ori[1] / point_ori[2], i_ori = point_ori[0] / point_ori[2];

      // Figure out the 3d position of the point
      cv::Vec3f pos = T_to3d * cv::Vec3f(i_ori, j_ori, 1);
      // Do not forget to re-scale in millimeters
      depth(j, i) = -pos[2]*1000;

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
  rect = cv::Rect(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);

  depth(rect).copyTo(depth_out);
  mask(rect).copyTo(mask_out);
}

void Renderer2d::renderImageOnly(cv::Mat &image_out, const cv::Rect &rect) const {
  // Figure out the transform from an original image pixel to a projected pixel
  // original frame: 0 is the top left corner of the pattern, X goes right, Y down, Z away from the camera
  // projected frame: 0 is the center of the projection image, X goes right, Y up, Z towards the camera

  // Scale the image properly
  float s = physical_width_ / img_ori_.cols;
  cv::Matx44f T_img_physical = cv::Matx44f(s,0,0,0, 0,s,0,0, 0,0,s,0, 0,0,0,1);

  // Flip axes and center at 0,0,0
  float physical_height = img_ori_.rows * s;
  T_img_physical = cv::Matx44f(1,0,0,-float(physical_width_) / 2, 0,1,0,float(physical_height) / 2, 0,0,1,0, 0,0,0,1) *
      cv::Matx44f(1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1) * T_img_physical;

  // Define the perspective transform to apply to the image (z=0 so we can ignore the 3rd column of P)
  cv::Matx34f P_noK = cv::Matx34f(R_(0, 0), R_(0, 1), R_(0, 2), T_(0), R_(1, 0), R_(1, 1), R_(1, 2), T_(1), R_(2, 0),
      R_(2, 1), R_(2, 2), T_(2)) * T_img_physical;
  cv::Matx33f T_to3d = cv::Matx33f(P_noK(0, 0), P_noK(0, 1), P_noK(0, 3), P_noK(1, 0), P_noK(1, 1), P_noK(1, 3),
      P_noK(2, 0), P_noK(2, 1), P_noK(2, 3));

  // Apply the camera transform
  cv::Matx33f T_img = K_ * T_to3d;

  // And readapt to an OpenCV image
  T_img = cv::Matx33f(1, 0, 0, 0, -1, 0, 0, 0, 1) * T_img;

  // Define the image corners
  std::vector<cv::Vec2f> corners(4);
  corners[0] = cv::Vec2f(0, 0);
  corners[1] = cv::Vec2f(img_ori_.cols, 0);
  corners[2] = cv::Vec2f(img_ori_.cols, img_ori_.rows);
  corners[3] = cv::Vec2f(0, corners[2][1]);

  // Project the image corners
  float x_min = std::numeric_limits<float>::max(), y_min = x_min;

  for (size_t i = 0; i < corners.size(); ++i) {
    cv::Vec3f res = T_img * cv::Vec3f(corners[i][0], corners[i][1], 1);
    x_min = std::min(res[0] / res[2], x_min);
    y_min = std::min(res[1] / res[2], y_min);
  }

  // Warp the mask
  cv::Size final_size(width_, height_);
  cv::Mat_<uchar> mask;
  T_img = cv::Matx33f(1, 0, -x_min, 0, 1, -y_min, 0, 0, 1) * T_img;

  // Compute the inverse
  cv::Matx33f T_img_inv = T_img.inv();

  cv::warpPerspective(mask_ori_, mask, T_img, final_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

  // Warp the image/depth
  cv::Mat_<cv::Vec3b> image = cv::Mat_<cv::Vec3b>::zeros(final_size);
  cv::Mat_<uchar>::iterator mask_it = mask.begin(), mask_end = mask.end();

  for (unsigned int j = 0; j < height_; ++j)
    for (unsigned int i = 0; i < width_; ++i, ++mask_it) {
      if (!*mask_it)
        continue;
      // Figure out the coordinates of the original point
      cv::Vec3f point_ori = T_img_inv * cv::Vec3f(i, j, 1);

      int j_ori = point_ori[1] / point_ori[2], i_ori = point_ori[0] / point_ori[2];
      image(j, i) = img_ori_(j_ori, i_ori);
    }

  image(rect).copyTo(image_out);
}

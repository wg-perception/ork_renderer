/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
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

#include <iostream>
#include <stdlib.h>

#include <GL/gl.h>

#include "model.h"

Renderer::Renderer(const std::string & mesh_path)
    :
      mesh_path_(mesh_path),
      angle_(0),
      width_(0),
      height_(0),
      focal_length_x_(0),
      focal_length_y_(0),
      near_(0),
      far_(0),
      model_(new Model()),
      scene_list_(0)
{
  // get a handle to the predefined STDOUT log stream and attach
  // it to the logging system. It remains active for all further
  // calls to aiImportFile(Ex) and aiApplyPostProcessing.
  ai_stream_ = new aiLogStream(aiGetPredefinedLogStream(aiDefaultLogStream_STDOUT, NULL));
  aiAttachLogStream(ai_stream_);
}

Renderer::~Renderer()
{
  // We added a log stream to the library, it's our job to disable it
  // again. This will definitely release the last resources allocated
  // by Assimp.
  aiDetachAllLogStreams();
}

void
Renderer::set_parameters(size_t width, size_t height, double focal_length_x, double focal_length_y, double near,
                         double far)
{
  width_ = width;
  height_ = height;

  focal_length_x_ = focal_length_x;
  focal_length_y_ = focal_length_y;

  near_ = near;
  far_ = far;

  clean_buffers();

  // Initialize the OpenGL context
  set_parameters_low_level();

  model_->LoadModel(mesh_path_);

  // Initialize the environment
  glClearColor(0.f, 0.f, 0.f, 1.f);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0); // Uses default lighting parameters

  glEnable(GL_DEPTH_TEST);

  glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  glEnable(GL_NORMALIZE);

  glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

  // Initialize the projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  double fx = Renderer::focal_length_x_;
  double fy = Renderer::focal_length_y_;
  double fovy = 2 * atan(0.5 * height_ / fy) * 180 / CV_PI;
  double aspect = (width_ * fy) / (height_ * fx);

  // set perspective
  gluPerspective(fovy, aspect, near, far);
  glViewport(0, 0, width_, height_);
}

void
Renderer::lookAt(GLdouble x, GLdouble y, GLdouble z, GLdouble upx, GLdouble upy, GLdouble upz)
{
  bind_buffers();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  gluLookAt(x, y, z, 0, 0, 0, upx, upy, upz);

  // scale the whole asset to fit into our view frustum
  aiVector3D scene_min, scene_max, scene_center;
  model_->get_bounding_box(&scene_min, &scene_max);
  scene_center.x = (scene_min.x + scene_max.x) / 2.0f;
  scene_center.y = (scene_min.y + scene_max.y) / 2.0f;
  scene_center.z = (scene_min.z + scene_max.z) / 2.0f;

  // center the model
  glTranslatef(-scene_center.x, -scene_center.y, -scene_center.z);

  // if the display list has not been made yet, create a new one and
  // fill it with scene contents
  if (scene_list_ == 0)
  {
    scene_list_ = glGenLists(1);
    glNewList(scene_list_, GL_COMPILE);
    // now begin at the root node of the imported data and traverse
    // the scenegraph by multiplying subsequent local transforms
    // together on GL's matrix stack.
    model_->Draw();
    glEndList();
  }

  glCallList(scene_list_);

  cv::Mat image_out, depth_out, mask_out;
  Renderer::render(image_out, depth_out, mask_out);
}

void
Renderer::render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out) const
{
  // Create images to copy the buffers to
  cv::Mat_ < cv::Vec3b > image(height_, width_);
  cv::Mat_<float> depth(height_, width_);
  cv::Mat_ < uchar > mask = cv::Mat_ < uchar > ::zeros(cv::Size(width_, height_));

  glFlush();

  // Get data from the depth/image buffers
  bind_buffers();

  // Deal with the RGB image
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels(0, 0, width_, height_, GL_BGR, GL_UNSIGNED_BYTE, image.ptr());

  // Deal with the depth image
  glReadBuffer(GL_DEPTH_ATTACHMENT);
  glReadPixels(0, 0, width_, height_, GL_DEPTH_COMPONENT, GL_FLOAT, depth.ptr());

  float zNear = near_, zFar = far_;
  cv::Mat_<float>::iterator it = depth.begin(), end = depth.end();
  float max_allowed_z = zFar * 0.99;

  unsigned int i_min = width_, i_max = 0, j_min = height_, j_max = 0;
  for (unsigned int j = 0; j < height_; ++j)
    for (unsigned int i = 0; i < width_; ++i, ++it)
    {
      //need to undo the depth buffer mapping
      //http://olivers.posterous.com/linear-depth-in-glsl-for-real
      *it = 2 * zFar * zNear / (zFar + zNear - (zFar - zNear) * (2 * (*it) - 1));
      if (*it > max_allowed_z)
        *it = 0;
      else
      {
        mask(j, i) = 255;
        // Figure the inclusive bounding box of the mask
        if (j > j_max)
          j_max = j;
        else if (j < j_min)
          j_min = j;
        if (i > i_max)
          i_max = i;
        else if (i < i_min)
          i_min = i;
      }
    }

  // Rescale the depth to be in millimeters
  cv::Mat depth_scale(cv::Size(width_, height_), CV_16UC1);
  depth.convertTo(depth_scale, CV_16UC1, 1e3);

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

  depth_scale(rect).copyTo(depth_out);
  image(rect).copyTo(image_out);
  mask(rect).copyTo(mask_out);
}

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

void
RendererIterator::render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out)
{
  if (isDone())
    return;

  cv::Vec3d t, up;
  view_params(t, up);

  renderer_->lookAt(t(0), t(1), t(2), up(0), up(1), up(2));
  renderer_->render(image_out, depth_out, mask_out);
}

/**
 * @return the rotation of the mesh with respect to the current view point
 */
cv::Matx33d
RendererIterator::R() const
{
  cv::Vec3d t, up;
  view_params(t, up);

  cv::Vec3d y = t.cross(up);
  cv::Mat R_full = (cv::Mat_<double>(3, 3) << t(0), t(1), t(2), y(0), y(1), y(2), up(0), up(1), up(2));
  cv::Matx33d R = R_full;

  return R;
}

/**
 * @return the translation of the mesh with respect to the current view point
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
  // from http://www.xsi-blog.com/archives/115
  static float inc = CV_PI * (3 - sqrt(5));
  static float off = 2.0 / float(n_points_);
  float y = index_ * off - 1 + (off / 2);
  float r = sqrt(1 - y * y);
  float phi = index_ * inc;
  float x = cos(phi) * r;
  float z = sin(phi) * r;

  float lat = acos(z), lon;
  if ((abs(sin(lat)) < 1e-5) || (abs(y / sin(lat)) > 1))
    lon = 0;
  else
    lon = asin(y / sin(lat));

  x *= radius_; // * cos(lon) * sin(lat);
  y *= radius_; //float y = radius * sin(lon) * sin(lat);
  z *= radius_; //float z = radius * cos(lat);

  // Figure out the up vector
  float x_up = radius_ * cos(lon) * sin(lat - 1e-5) - x;
  float y_up = radius_ * sin(lon) * sin(lat - 1e-5) - y;
  float z_up = radius_ * cos(lat - 1e-5) - z;
  normalize_vector(x_up, y_up, z_up);

  // Figure out the third vector of the basis
  float x_right = -y_up * z + z_up * y;
  float y_right = x_up * z - z_up * x;
  float z_right = -x_up * y + y_up * x;
  normalize_vector(x_right, y_right, z_right);

  // Rotate the up vector in that basis
  float angle_rad = angle_ * CV_PI / 180.;
  float x_new_up = x_up * cos(angle_rad) + x_right * sin(angle_rad);
  float y_new_up = y_up * cos(angle_rad) + y_right * sin(angle_rad);
  float z_new_up = z_up * cos(angle_rad) + z_right * sin(angle_rad);

  T = cv::Vec3d(x, y, z);
  up = cv::Vec3d(x_new_up, y_new_up, z_new_up);
}

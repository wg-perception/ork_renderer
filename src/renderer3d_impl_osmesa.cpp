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

#include "renderer3d_impl_osmesa.h"

#include <cstdlib>
#include <cstring>

/**
 * @param file_path the path of the mesh file
 */
Renderer3dImpl::Renderer3dImpl(const std::string & mesh_path, int width, int height)
    :
      Renderer3dImplBase(mesh_path, width, height),
      ctx_(0),
      ctx_buffer_(0)
{
}

void
Renderer3dImpl::clean_buffers()
{
  if (ctx_) {
    OSMesaDestroyContext(ctx_);
    ctx_ = 0;
  }

  if (ctx_buffer_)
  {
    free(ctx_buffer_);
    ctx_buffer_ = 0;
  }
}

void
Renderer3dImpl::set_parameters_low_level()
{
  ctx_ = OSMesaCreateContextExt(OSMESA_RGB, 16, 0, 0, NULL);

  ctx_buffer_ = malloc(width_ * height_ * 3 * sizeof(GLubyte));
  OSMesaMakeCurrent(ctx_, ctx_buffer_, GL_UNSIGNED_BYTE, width_, height_);
}

void
Renderer3dImpl::bind_buffers() const
{
}

void
Renderer3dImpl::get_buffers(int width, int height, void* rgb, void* depth) const
{
  GLint tmp_width;
  GLint tmp_height;
  GLint tmp_format;
  void* tmp;

  if (rgb) {
    OSMesaGetColorBuffer( ctx_, &tmp_width, &tmp_height, &tmp_format, &tmp );
    std::memcpy(rgb, tmp, tmp_width*tmp_height*tmp_format);
  }

  if (depth) {
    OSMesaGetDepthBuffer( ctx_, &tmp_width, &tmp_height, &tmp_format, &tmp );
    std::memcpy(depth, tmp, tmp_width*tmp_height*tmp_format);
  }
}

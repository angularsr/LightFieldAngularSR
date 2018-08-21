// @file im2row_cpu.cpp
// @brief Stack image patches as matrix rows (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2row6D.hpp"
#include <string.h>
#include <iostream>

using namespace vl ;
using namespace vl::impl ;

/* ---------------------------------------------------------------- */
/*                                                  Heper functions */
/* ---------------------------------------------------------------- */

static inline int floor_divide(int a, int b) {
  if (a >= 0) return a/b;
  else return (a - b + 1)/b;
}

static inline int ceil_divide(int a, int b) {
  if (a >= 0) return (a + b - 1)/b ;
  else return a/b ;
}

static inline int static_max(int a, int b) {
  return (a>=b) ? a:b ;
}

static inline int static_min(int a, int b) {
  return (a<=b) ? a:b ;
}

namespace vl { namespace impl {


  template<typename type>
  struct im2row6D<vl::VLDT_CPU, type>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    forward(Context & context,
            type* stacked,
            type const* data,
            size_t width,
            size_t height,
            size_t heightAn,
            size_t widthAn,
            size_t depth,
            size_t windowWidth,
            size_t windowHeight,
            size_t windowWidthAn,
            size_t windowHeightAn,
            size_t strideX,
            size_t strideY,
            size_t strideXAn,
            size_t strideYAn,
            size_t padLeft,
            size_t padRight,
            size_t padTop,
            size_t padBottom,
            size_t padLeftAn,
            size_t padRightAn,
            size_t padTopAn,
            size_t padBottomAn)

    {

      int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
      int numPatchesXAn = (widthAn + (padLeftAn + padRightAn) - windowWidthAn)/strideXAn + 1 ;
      int numPatchesYAn = (heightAn + (padTopAn + padBottomAn) - windowHeightAn)/strideYAn + 1 ;
      int numRows = windowWidth * windowHeight * windowWidthAn * windowHeightAn * depth ;

      /*
       Fill a row of the patch matrix. Since patches are stored
       along the columns of the matrix, scanning a row menas visiting all
       the patches. Different rows corresponds to a different
       offset within each patch.

       In this manner, as we fill a row
       we tend to access spatially adiacent elements
       in the input image, particulary for small strides.
       */
      for (int row = 0; row < numRows ; ++row) {
        /*
         Get the patch offset corresponding to this row of the stacked
         image.
         */
        int u = row ;
        int v = u / windowWidth ;
        int uAn = v / windowHeight ;
        int vAn = uAn / windowWidthAn ;
        int z = vAn / windowHeightAn ;
        u %= windowWidth ;
        v %= windowHeight ;
        uAn %= windowWidthAn ;
        vAn %= windowHeightAn ;

        /*
         Filling this row requires visiting the pixels in the input tensor
         `data` that appear at the given offset (u,v) in the output patches.
         For the patch at (x,y), the pixel coordinates (x_data,y_data) in the
         `data` tensor are:

         x_data(x) = x * strideX + u * dilateX - padLeft,  0 <= x < numPatchesX,
         y_data(y) = y * strideY + v * dilateY - padTop,   0 <= y < numPatchesY,
         z_data(z) = z.

         Now we visit all patches (x,y) in lexicographical order to fill
         successive output pixels. Patches around the boundary may peek outside
         the `data` tensor, which is padded with zero. We calcualte these
         borders here and fill them with zeros in the output.
         
         In particular, patch x peeks within the input tensor `data`
         if x is in the range [x0,x1] given by:

         x_data(x) >= 0
         <=> x >= (padLeft - u * dilateX) / stride
         <=> x >= ceil((padLeft - u * dilateX) / stride) = x0
         
         x_data(x) <= width-1
         <=> x <= (width-1 + padLeft - u * dilateX) / stride
         <=> x <= floor((width-1 + padLeft - u * dilateX) / stride)
         <=> x <  floor((width-1 + padLeft - u * dilateX) / stride) + 1 = x1

         and the same for y. Note that, while usually x0 <= x1, there are
         special cases for which x1 < x0. This is accounted for in the loops
         below.
         */

        int x0 = static_min(numPatchesX, ceil_divide(padLeft - u, strideX)) ;
        int y0 = static_min(numPatchesY, ceil_divide(padTop - v, strideY)) ;
        int x1 = static_min(numPatchesX, floor_divide(width-1 + padLeft - u, strideX) + 1) ;
        int y1 = static_min(numPatchesY, floor_divide(height-1 + padTop - v, strideY) + 1) ;
        int x0An = static_min(numPatchesXAn, ceil_divide(padLeftAn - uAn, strideXAn)) ;
        int y0An = static_min(numPatchesYAn, ceil_divide(padTopAn - vAn, strideYAn)) ;
        int x1An = static_min(numPatchesXAn, floor_divide(widthAn-1 + padLeftAn - uAn, strideXAn) + 1) ;
        int y1An = static_min(numPatchesYAn, floor_divide(heightAn-1 + padTopAn - vAn, strideYAn) + 1) ;
        int x ;
        int y ;
        int xAn ;
        int yAn ;

        /* Inner xAn, YAn, x, and Y */
        for (yAn = 0 ; yAn < y0An ; ++yAn) {
          for (xAn = 0 ; xAn < numPatchesXAn ; ++xAn) {
            for (y = 0 ; y < numPatchesY ; ++y) {
              for (x = 0 ; x < numPatchesX ; ++x) {
              *stacked++ = 0 ;
              }
            }
          }
        }

                
        for (; yAn < y1An ; ++yAn) {
          /* Inner xAn, x and y */
          for (xAn = 0 ; xAn < x0An ; ++xAn) {
            for (y = 0 ; y < numPatchesY ; ++y) {
              for (x = 0 ; x < numPatchesX ; ++x) {
              *stacked++ = 0 ;
              }
            }
          }

          for (; xAn < x1An ; ++xAn) {
	  /*  Inner x and y */
	    for (y = 0 ; y < y0 ; ++y) {
	      for (x = 0 ; x < numPatchesX ; ++x) {
	        *stacked++ = 0 ;
	      }
	    }
	    for ( ; y < y1 ; ++y) {
	      /*  Inner x */
	      for (x = 0 ; x < x0 ; ++x) {
	        *stacked++ = 0 ;
	      }
	      int y_data = y * strideY + v - padTop ;
	      int x_data = x * strideX + u - padLeft ;
	      int yAn_data = yAn * strideYAn + vAn - padTopAn ;
	      int xAn_data = xAn * strideXAn + uAn - padLeftAn ;
	      type const * b = data + ( ( ( z * heightAn + yAn_data ) * widthAn + xAn_data ) * height + y_data) * width + x_data ;
	      for ( ; x < x1 ; ++x) {
	        *stacked++ = *b ;
	        b += strideX ;
	      }
	      for ( ; x < numPatchesX ; ++x) {
	        *stacked++ = 0 ;
	      }
	    }
	    for ( ; y < numPatchesY ; ++y) {
	      for (x = 0 ; x < numPatchesX ; ++x) {
	        *stacked++ = 0 ;
	      }
	    }
          }

          for ( ; xAn < numPatchesXAn ; ++xAn) {
            for (y = 0 ; y < numPatchesY ; ++y) {
              for (x = 0 ; x < numPatchesX ; ++x) {
                *stacked++ = 0 ;
              }
            }
          }
        }

        for ( ; yAn < numPatchesYAn ; ++yAn) {
          for (xAn = 0 ; xAn < numPatchesXAn ; ++xAn) {
            for (y = 0 ; y < numPatchesY ; ++y) {
              for (x = 0 ; x < numPatchesX ; ++x) {
              *stacked++ = 0 ;
              }
            }
          }
        }

      }
      return vl::VLE_Success ;
    }

    /* ------------------------------------------------------------ */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward(Context & context,
             type* data,
             type const* stacked,
             size_t width,
             size_t height,
             size_t heightAn,
             size_t widthAn,
             size_t depth,
             size_t windowWidth,
             size_t windowHeight,
             size_t windowWidthAn,
             size_t windowHeightAn,
             size_t strideX,
             size_t strideY,
             size_t strideXAn,
             size_t strideYAn,
             size_t padLeft,
             size_t padRight,
             size_t padTop,
             size_t padBottom,
             size_t padLeftAn,
             size_t padRightAn,
             size_t padTopAn,
             size_t padBottomAn)
    {

      int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
      int numPatchesXAn = (widthAn + (padLeftAn + padRightAn) - windowWidthAn)/strideXAn + 1 ;
      int numPatchesYAn = (heightAn + (padTopAn + padBottomAn) - windowHeightAn)/strideYAn + 1 ;
      int numRows = windowWidth * windowHeight * windowWidthAn * windowHeightAn * depth ;

      memset(data, 0, sizeof(type) * width * height * widthAn * heightAn * depth) ;

      /*
       Do the converse of im2col, still scanning rows of the stacked image.
       See comments of im2col for an explanation of the algorithm.
       */
      for (int row = 0; row < numRows ; ++row) {
        int u = row ;
        int v = u / windowWidth ;
        int uAn = v / windowHeight ;
        int vAn = uAn / windowWidthAn ;
        int z = vAn / windowHeightAn ;
        u %= windowWidth ;
        v %= windowHeight ;
        uAn %= windowWidthAn ;
        vAn %= windowHeightAn ;

        int x0 = static_min(numPatchesX, ceil_divide(padLeft - u, strideX)) ;
        int y0 = static_min(numPatchesY, ceil_divide(padTop - v, strideY)) ;
        int x1 = static_min(numPatchesX, floor_divide(width-1 + padLeft - u, strideX) + 1) ;
        int y1 = static_min(numPatchesY, floor_divide(height-1 + padTop - v, strideY) + 1) ;
        int x0An = static_min(numPatchesXAn, ceil_divide(padLeftAn - uAn, strideXAn)) ;
        int y0An = static_min(numPatchesYAn, ceil_divide(padTopAn - vAn, strideYAn)) ;
        int x1An = static_min(numPatchesXAn, floor_divide(widthAn-1 + padLeftAn - uAn, strideXAn) + 1) ;
        int y1An = static_min(numPatchesYAn, floor_divide(heightAn-1 + padTopAn - vAn, strideYAn) + 1) ;
        int x ;
        int y ;
        int xAn ;
        int yAn ;

        yAn = static_max(0, y0An) ;
        stacked += numPatchesXAn * numPatchesY * numPatchesX * static_max(yAn, 0) ;

        for ( ; yAn < y1An ; ++yAn) {

          xAn = static_max(0, x0An) ;
          stacked += numPatchesY * numPatchesX * static_max(xAn, 0) ;

          for ( ; xAn < x1An ; ++xAn) {
            y = static_max(0, y0) ;
            stacked += numPatchesX * static_max(y, 0) ;

            for ( ; y < y1 ; ++y) {
              x = static_max(0, x0) ;
              int y_data = y * strideY + v - padTop ;
              int x_data = x * strideX + u - padLeft ;
              int yAn_data = yAn * strideYAn + vAn - padTopAn ;
              int xAn_data = xAn * strideXAn + uAn - padLeftAn ;

              type * b = data + ( ( ( z * heightAn + yAn_data ) * widthAn + xAn_data ) * height + y_data) * width + x_data ;
              stacked += x ;
              for ( ; x < x1 ; ++x) {
                *b += *stacked++ ;
                b += strideX ;
              }
              stacked += numPatchesX - x ;
            }
            stacked += numPatchesX * (numPatchesY - y) ;
          }
          stacked += numPatchesX * numPatchesY * ( numPatchesXAn - xAn );
        }
        stacked += numPatchesX * numPatchesY * numPatchesXAn * ( numPatchesYAn - yAn );
      }
      return vl::VLE_Success ;
    }
  } ;

} }

// Instantiations
template struct vl::impl::im2row6D<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::im2row6D<vl::VLDT_CPU, double> ;
#endif

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "timer.h"

#define ITERS 100

using namespace cv;
using namespace std;

void compute(gpu::GpuMat &in, gpu::GpuMat &bgr, gpu::GpuMat &out)
{
  cv::gpu::demosaicing(in, bgr, cv::COLOR_BayerBG2BGR);
  cv::gpu::resize(bgr, out, out.size());
}

int main(void)
{
  int w = 4608;
  int h = 3288;
  int wnew = 800;
  int hnew = 600;

  Mat in(h, w, CV_8UC1);
  Mat out(hnew, wnew, CV_8UC3);
  gpu::GpuMat d_in;
  gpu::GpuMat d_bgr(h, w, CV_8UC3);
  gpu::GpuMat d_out(hnew, wnew, CV_8UC3);

  double t = GetRealTime();
  for (int i = 0; i < ITERS; i++)
  {
    in.setTo(i);
    d_in.upload(in);
    compute(d_in, d_bgr, d_out);
    d_out.download(out);
  }
  cout << "Old Time: " << GetRealTime()-t << " (" << cv::sum(out)[0] << ")" << endl;

  gpu::CudaMem c_in(h, w, CV_8UC1, gpu::CudaMem::ALLOC_ZEROCOPY);
  gpu::CudaMem c_out(hnew, wnew, CV_8UC3, gpu::CudaMem::ALLOC_ZEROCOPY);
  d_in = c_in.createGpuMatHeader();
  d_out = c_out.createGpuMatHeader();
  out = c_out.createMatHeader();

  t = GetRealTime();
  for (int i = 0; i < ITERS; i++)
  {
    d_in.setTo(i);
    compute(d_in, d_bgr, d_out);
  }
  cout << "New Time: " << GetRealTime()-t << " (" << cv::sum(out)[0] << ")" << endl;
  
    
  return 0;
}

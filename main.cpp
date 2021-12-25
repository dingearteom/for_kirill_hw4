#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl.h>
#endif
#include "cl2.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      // for (auto platform : platforms) {
      //    std::string name;
      //    platform.getInfo(CL_PLATFORM_NAME, &name);
      //    std::cout << name << std::endl;
      // }
      platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      // load opencl source
      std::ifstream cl_file("bitonic.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, cl_string);

      // create program
      cl::Program program(context, source);

      // compile opencl source
      program.build(devices);

      std::ifstream input_file("input.txt");
      std::ofstream output_file("output.txt");

      size_t N;
      size_t M;
      input_file >> N;
      M = ceil(log2(N));

      std::vector<float> as(N);
      for (size_t i = 0; i < N; i++) {
         input_file >> as[i];
      }

      // allocate device buffer to hold message
      cl::Buffer as_gpu (context, CL_MEM_READ_ONLY, sizeof(int) * N);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(as_gpu, CL_TRUE, 0, sizeof(int) * N, &as[0]);

      // load named kernel from opencl source
      size_t workgroup_size = 128;
      size_t global_workgroup_size = (N / 2 + workgroup_size - 1) / workgroup_size * workgroup_size;
      cl::Kernel kernel(program, "bitonic");
      cl::KernelFunctor<cl::Buffer, int, int, int> bitonic(kernel);
      cl::EnqueueArgs eargs(queue, cl::NDRange(global_workgroup_size), cl::NDRange(workgroup_size));

      for (int k = 1; k <= M; k++) { // size of block
            for (int t = k - 1; t >= 0; t--) { // size of arrow
               cl::Event event = bitonic(eargs, as_gpu, (int) N, k, t);
               event.wait();
            }
      }

      queue.enqueueReadBuffer(as_gpu, CL_TRUE, 0, sizeof(int) * N, &as[0]);
      for (size_t i = 0; i < N; ++i) {
         output_file << as[i];
         if (i != N - 1) {
            output_file << " ";
         }
      }

   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}

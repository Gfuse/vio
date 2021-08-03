//
// Created by root on 4/27/21.
//
#include "vio/cl_class.h"
opencl::opencl() {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0){
        std::cerr << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    for(auto i:all_platforms){
        std::cout << "Find platform number:"<< i.getInfo<CL_PLATFORM_NAME>() << "\n";
        i.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    }

    if (all_devices.size() == 0){
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    device = all_devices[0];
    std::cout << "CL_DEVICE_NAME: " << device.getInfo<CL_DEVICE_NAME>() <<'\n'
            << "CL_DEVICE_OPENCL_C_VERSION: " <<device.getInfo<CL_DEVICE_OPENCL_C_VERSION>()<<'\n'
            << "CL_DEVICE_BUILT_IN_KERNELS: " <<device.getInfo<CL_DEVICE_BUILT_IN_KERNELS>()<<'\n'
            << "CL_DEVICE_COMPILER_AVAILABLE: " <<device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>()<<'\n'
            << "CL_DEVICE_LOCAL_MEM_SIZE: " <<device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()<<'\n'
             << "CL_DEVICE_GLOBAL_MEM_SIZE: " <<device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()<<'\n';

    context=new cl::Context({ device });
    cl::Program::Sources sources;
    read_cl fast(std::string(KERNEL_DIR)+"/fast-gray.cl");
    sources.push_back({ fast.src_str, fast.size });
    read_cl compute_residual(std::string(KERNEL_DIR)+"/compute-residual.cl");
    sources.push_back({ compute_residual.src_str, compute_residual.size });
    program=new cl::Program(*context, sources);
    if(program->build({ device },"-DFAST_THRESH=40 -DPATCH_SIZE=4 -DPATCH_HALFSIZE=2") !=0)
        std::cout << " Error building: " << program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)<<'\n'
        <<program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
    queue=new cl::CommandQueue(*context,device,0,NULL);
}
opencl::~opencl() {
}
void opencl::clear_buf() {
    queue->finish();
    queue->flush();
    _kernels.clear();
}


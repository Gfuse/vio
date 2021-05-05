//
// Created by root on 4/27/21.
//
#include "svo/cl_class.h"
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
        std::cout << "Find platform: " << i.getInfo<CL_PLATFORM_NAME>() << "\n";
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
    read_cl int_add("../kernel/int_add.cl");
    sources.push_back({ int_add.src_str, int_add.size });
    read_cl double_add("../kernel/double_add.cl");
    sources.push_back({ double_add.src_str, double_add.size });
    read_cl corner_10_less("../kernel/is_corner_10_less.cl");
    sources.push_back({ corner_10_less.src_str, corner_10_less.size });
    read_cl prefast("../kernel/prefast-gray.cl");
    sources.push_back({ prefast.src_str, prefast.size });
    //read_cl fast("../kernel/fast-gray.cl");
   // sources.push_back({ fast.src_str, fast.size});
    program=new cl::Program(*context, sources);
    if(program->build({ device },"-DFAST_THRESH=40 -DFAST_RING=15") !=0)
        std::cout << " Error building: " << program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)<<'\n'
        <<program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
    queue=new cl::CommandQueue(*context,device,0,NULL);
}
opencl::~opencl() {
    queue->finish();
}
void opencl::clear_buf() {
    _kernels.clear();
    queue->flush();
}


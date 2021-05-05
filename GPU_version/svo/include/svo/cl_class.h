//
// Created by root on 4/27/21.
//

#ifndef VIO_OPENCL_CL_CLASS_H
#define VIO_OPENCL_CL_CLASS_H
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <exception>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/opencl.h>
#include <opencv2/opencv.hpp>

class kernel{
public:
    kernel(cl::Program* program,std::string name){_kernel = new cl::Kernel(*program,name.c_str());};
    ~kernel(){
        //for(auto i:_buffers)i.first.first.;
        _buffers.clear();
        _images.clear();
    };
    template<typename T>
    int32_t write(size_t id/*buffer ID*/,T* buf,cl::CommandQueue* queue,cl::Context* context,size_t buf_size){
        try{
            _buffers.push_back(std::pair<std::pair<cl::Buffer,size_t>,size_t>(std::pair<cl::Buffer,size_t>(cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(T) * buf_size),
                                                                                                           buf_size),id));
            queue->enqueueWriteBuffer(_buffers.back().first.first,CL_TRUE, 0, sizeof(T) * buf_size, buf);
            _kernel->setArg(id,_buffers.back().first.first);
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Error C:Kernel, F:write\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    };
    int32_t write(size_t id/*buffer ID*/,cv::Mat& buf,cl::Context* context){
            try{
                //if(id==0){
                    _images.push_back(std::pair<cl::Image2D,size_t>(cl::Image2D(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                                                cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                                                                buf.size().width,
                                                                                buf.size().height,
                                                                                0,
                                                                                reinterpret_cast<uchar*>(buf.data)),id));
               /* }else{
                    _images.push_back(std::pair<cl::Image2D,size_t>(cl::Image2D(*context, CL_MEM_READ_WRITE,
                                                                                cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
                                                                                buf.size().width,
                                                                                buf.size().height),id));
                }*/

                _kernel->setArg(id,_images.back().first);
                return CL_SUCCESS;
            }catch (std::exception& err){
                std::cerr<<"Error C:Kernel, F:write image\n"<<err.what()<<'\n';
                return CL_MAP_FAILURE;
            };
    };

    template<typename T>
    int32_t reload(size_t id,T* buf,cl::CommandQueue* queue){
        try{
            if(id>_buffers.size())return CL_MAP_FAILURE;
            for(auto i:_buffers)if(i.second==id)
                queue->enqueueWriteBuffer(i.first.first,CL_TRUE, 0, sizeof(T) * i.first.second, buf);
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Reload buffer on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    };
    int32_t reload(size_t id,cv::Mat& buf,cl::Context* context){
        try{
            for(auto i:_images)if(i.second==id)
                    i.first=cl::Image2D(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                        buf.size().width,
                                        buf.size().height,
                                        0,
                                        reinterpret_cast<uchar*>(buf.data));
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Reload buffer on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    };
    cl::Kernel* _kernel = NULL;
private:
    std::vector<std::pair<std::pair<cl::Buffer,size_t>,size_t>> _buffers;
    std::vector<std::pair<cl::Image2D,size_t>> _images;

};
class read_cl{
public:
    read_cl(std::string path){
        src_str=(char *)malloc(0x100000);
        FILE *fp;
        fp = fopen(path.c_str(), "r");
        size = fread(src_str, 1, 0x100000, fp);
        fclose(fp);
    }
    void print(){
        std::cerr<<src_str<<"\n";
    }
    char *src_str = nullptr;
    size_t size;
};
class opencl{
public:
    opencl();
    ~opencl();
    int32_t make_kernel(std::string name){_kernels.push_back(kernel(program,name));};
    template<typename T>
    int32_t write_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/, int size,T* buf) {
        try{
            _kernels.at(id1).write(id2,buf,queue,context,size);
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Write buffer on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    }
    int32_t write_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,cv::Mat& buf) {
        try{
            _kernels.at(id1).write(id2,buf,context);
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Write buffer on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    }
    template<typename T>
    int32_t write_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,T& value){
        _kernels.at(id1)._kernel->setArg(id2,value);
    };
    template<typename T>
    int32_t reload_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,T* buf) {
        try{
            _kernels.at(id1).reload(id2,buf,queue);
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Reload buffer on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    }
    int32_t reload_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,cv::Mat& buf) {
        try{
            _kernels.at(id1).reload(id2,buf,context);
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Write buffer on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    }
    template<typename T>
    int32_t run(size_t id1/*kernal ID*/,std::size_t  x=1,std::size_t y=1,std::size_t z=1,size_t id2=0/*last id of the kernel related to the output*/,
                size_t size_out=0/*out put buffer size*/,
                T* out= nullptr) {
        try{
            //std::cerr<<"kernel: "<<_kernels.at(id1)._kernel->getInfo<CL_KERNEL_FUNCTION_NAME>();
            //std::cerr<<" ,Args: "<<_kernels.at(id1)._kernel->getInfo<CL_KERNEL_NUM_ARGS>()<<" ,Run out: ";
            cl::Buffer out_buf(*context, CL_MEM_READ_WRITE, sizeof(T) * size_out);
            _kernels.at(id1)._kernel->setArg(id2,out_buf);
            if(z>1 && y>1){
                std::cerr<<queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y,z)/*Global*/, cl::NullRange/*local*/)<<'\n';
            }else if(z<2 && y>1){
                queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y)/*Global*/, cl::NullRange/*local*/);
                //std::cerr<<queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y)/*Global*/, cl::NullRange/*local*/)<<'\n';
                /*std::cerr<<_kernels.at(id1)._kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(0)<<'\n';
                std::cerr<<_kernels.at(id1)._kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(1)<<'\n';
                std::cerr<<_kernels.at(id1)._kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(2)<<'\n';
                std::cerr<<_kernels.at(id1)._kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(3)<<'\n';*/
            }else{
                std::cerr<<queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x)/*Global*/, cl::NullRange/*local*/)<<'\n';
            };
            queue->enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(T) * size_out, out);
            //wait for kernel to finish
            queue->finish();
            return CL_SUCCESS;
        } catch (std::exception& err) {
            std::cerr<<"RUN kernel Error\n"<<err.what()<<'\n';
            return CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
        }

    }
    void clear_buf();
private:
    std::vector<kernel> _kernels;
    cl::Context* context = NULL;
    cl::Device device;
    cl::Program* program = NULL;
    cl::CommandQueue* queue=NULL;
};

#endif //VIO_OPENCL_CL_CLASS_H

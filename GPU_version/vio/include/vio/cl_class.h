//
// Created by root on 4/27/21.
//

#ifndef VIO_OPENCL_CL_CLASS_H
#define VIO_OPENCL_CL_CLASS_H
#include <vio/cl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <exception>
#include <CL/opencl.h>
#include <opencv2/opencv.hpp>
#include <vio/abstract_camera.h>

class kernel{
public:
    kernel(cl::Program* program,std::string name){_kernel = new cl::Kernel(*program,name.c_str());};
    ~kernel(){
        for(auto&& i:_images)i.first.setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
        for(auto&& i:_buffers)i.first.first.setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
        _buffers.clear();
        _images.clear();
    };
    template<typename T>
    int32_t write(size_t id/*buffer ID*/,T* buf,cl::CommandQueue* queue,cl::Context* context,size_t buf_size){
        _buffers.push_back(std::pair<std::pair<cl::Buffer,size_t>,size_t>(std::pair<cl::Buffer,size_t>(cl::Buffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR
                                                                                                                  , sizeof(T) * buf_size,buf),
                                                                                                       buf_size),id));
        cl_int err=_kernel->setArg(id,_buffers.back().first.first);
        if(err!=0){
            std::cerr<<"Error C:Kernel, F:write, L3:setArg\t"<<id<<"\t"<<err<<'\n';
            return err;
        }
        return CL_SUCCESS;

    };
    int32_t write(size_t id/*buffer ID*/,cv::Mat& buf,cl::Context* context){
        _images.push_back(std::pair<cl::Image2D,size_t>(cl::Image2D(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                                                    cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                                                    buf.size().width,
                                                                    buf.size().height,
                                                                    0,
                                                                    reinterpret_cast<uchar*>(buf.data)),id));

        cl_int err=_kernel->setArg(id,_images.back().first);
        if(err!=0){
            std::cerr<<"Error C:Kernel, F:write image, L:setArg ID:\t"<<id<<"\t"<<err<<'\n';
            return err;
        }
        return CL_SUCCESS;
    };

    template<typename T>
    int32_t reload(size_t id,T* buf,cl::CommandQueue* queue){
        for(auto&& i:_buffers)if(i.second==id){
            T* Map_buf=(T*)queue->enqueueMapBuffer(i.first.first,CL_TRUE,CL_MAP_WRITE,0,sizeof(T) * i.first.second);
            std::cerr<<"\n reload class L:59,\t";
            memcpy(Map_buf,buf,sizeof(T) * i.first.second);
             std::cerr<<"L:61\t";
            if(queue->enqueueUnmapMemObject(i.first.first,Map_buf)!=CL_SUCCESS)std::cerr<<"Reload buffer on GPU failed"<<'\n';
            std::cerr<<"L:63\n";
        }
        return CL_SUCCESS;
    };
    int32_t reload(size_t id,cv::Mat& buf,cl::Context* context){
        try{
            for(auto&& i:_images)if(i.second==id){
                i.first.setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
                    i.first=cl::Image2D(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                        buf.size().width,
                                        buf.size().height,
                                        0,
                                        reinterpret_cast<uchar*>(buf.data));
                    cl_int err=_kernel->setArg(id,i.first);
                    if(err!=0){
                        std::cerr<<"Error C:Kernel, F:write image, L:setArg ID:\t"<<id<<"\t"<<err<<'\n';
                        return err;
                    }
                }
            return CL_SUCCESS;
        }catch (std::exception& err){
            std::cerr<<"Reload buffer on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    };
    cl::Kernel* _kernel = NULL;
    std::pair<cl::Buffer,size_t> read(size_t id){
        for(auto i:_buffers)if(i.second==id)return i.first;
    }
    cl::Image2D read_img(size_t id){
        for(auto i:_images)if(i.second==id)return i.first;
    }
private:
    static void notify(cl_mem *, void * user_data) {
        //std::cerr << "Memory object was deleted." << std::endl;
    }
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
    opencl(vk::AbstractCamera* cam);
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
            std::cerr<<"Write image on GPU failed\n"<<err.what()<<'\n';
            return CL_MAP_FAILURE;
        }
    }
    template<typename T>
    int32_t write_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,T&& value){
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
    int32_t run(size_t id1/*kernal ID*/,std::size_t  x=1,std::size_t y=1,std::size_t z=1) {
        cl_int err=0;
        queue->flush();
        if(z>1 && y>1){
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y,z)/*Global*/, cl::NullRange/*local*/);
        }else if(z<2 && y>1){
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y)/*Global*/, cl::NullRange/*local*/);
        }else{
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x)/*Global*/, cl::NullRange/*local*/);
        };
        if(err!=0){
            std::cerr<<"kernel: "<<_kernels.at(id1)._kernel->getInfo<CL_KERNEL_FUNCTION_NAME>();
            std::cerr<<" ,Args: "<<_kernels.at(id1)._kernel->getInfo<CL_KERNEL_NUM_ARGS>()<<" ,Run out: "<<err<<'\n';
            for(int i=0;i<_kernels.at(id1)._kernel->getInfo<CL_KERNEL_NUM_ARGS>();++i)
                std::cerr<<"argument type "<<i<<":"<<_kernels.at(id1)._kernel->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(i)<<'\n';
        }
        return queue->finish();

    }
    template<typename T>
    void read(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/, T* out){
        queue->enqueueReadBuffer(_kernels.at(id1).read(id2).first, CL_TRUE, 0, sizeof(T) * _kernels.at(id1).read(id2).second, out);
    }
    template<typename T>
    void read(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,size_t size/*size*/, T* out){
        std::cerr<<"read GPU line 191\t";
        T* Map_buf=(T*)queue->enqueueMapBuffer(_kernels.at(id1).read(id2).first,CL_TRUE,CL_MAP_READ,0,sizeof(T) * size);
        std::cerr<<"read GPU line 193\t";
        memcpy(out,Map_buf,sizeof(T) * size);
        std::cerr<<"read GPU line 195\t";
        if(queue->enqueueUnmapMemObject(_kernels.at(id1).read(id2).first,Map_buf)!=CL_SUCCESS)std::cerr<<"Read buffer on GPU failed"<<'\n';
        std::cerr<<"read GPU line 197\t";
    }
    void clear_buf();
private:
    std::vector<kernel> _kernels;
    cl::Context* context = nullptr;
    cl::Device* device = nullptr;
    cl::Program* program = nullptr;
    cl::CommandQueue* queue= nullptr;
    vk::AbstractCamera* cam= nullptr;
};

#endif //VIO_OPENCL_CL_CLASS_H

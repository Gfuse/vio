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
    kernel(cl::Program* program,std::string name){
        cl_int error;
        _kernel = new cl::Kernel(*program,name.c_str(),&error);
        assert(error== CL_SUCCESS);
    };
    ~kernel(){
        for(auto&& i:_images)i.first.setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
        for(auto&& i:_buffers)i.first.first.setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
        _buffers.clear();
        _images.clear();
    };
    template<typename T>
    int32_t write(size_t id/*buffer ID*/,T* buf,cl::CommandQueue* queue,cl::Context* context,size_t buf_size){
        assert(buf);
        cl_int error;
        _buffers.push_back(std::pair<std::pair<cl::Buffer,size_t>,size_t>(std::pair<cl::Buffer,size_t>(cl::Buffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR
                , sizeof(T) * buf_size,buf,&error),
                                                                                                       buf_size),id));
        cl_int err=_kernel->setArg(id,_buffers.back().first.first);
        assert(err == CL_SUCCESS);
        assert(error == CL_SUCCESS);
        return error;

    };
    int32_t write(size_t id/*buffer ID*/,cv::Mat& buf,cl::Context* context){
        assert(!buf.empty());
        cl_int error;
        _images.push_back(std::pair<cl::Image2D,size_t>(cl::Image2D(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                                                    cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                                                    buf.size().width,
                                                                    buf.size().height,
                                                                    0,
                                                                    reinterpret_cast<uchar*>(buf.data),&error),id));
        cl_int err=_kernel->setArg(id,_images.back().first);
        assert(err == CL_SUCCESS);
        assert(error == CL_SUCCESS);
        return error;
    };

    template<typename T>
    int32_t reload(size_t id,T* buf,cl::CommandQueue* queue){
        assert(buf);
        cl_int error;
        for(auto&& i:_buffers)if(i.second==id){
                cl::Event event;
                T* Map_buf=(T*)queue->enqueueMapBuffer(i.first.first,CL_NON_BLOCKING,CL_MAP_WRITE,0,sizeof(T) * i.first.second,NULL,&event,&error);
                event.wait();
                memcpy(Map_buf,buf,sizeof(T) * i.first.second);
                assert(queue->enqueueUnmapMemObject(i.first.first,Map_buf,NULL,&event)==CL_SUCCESS);
                event.wait();

        }
        assert(error == CL_SUCCESS);
        return error;
    };
    int32_t reload(size_t id,cv::Mat& buf,cl::Context* context){
        assert(!buf.empty());
        cl_int error;
        try{
            for(auto&& i:_images)if(i.second==id){
                    i.first.setDestructorCallback((void (*)(_cl_mem *, void *))notify, NULL);
                    i.first=cl::Image2D(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
                                        buf.size().width,
                                        buf.size().height,
                                        0,
                                        reinterpret_cast<uchar*>(buf.data),&error);
                    cl_int err=_kernel->setArg(id,i.first);
                    assert(err == CL_SUCCESS);
                }
            assert(error == CL_SUCCESS);
            return error;
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
    bool write_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/, int size,T* buf) {
            if(_kernels.at(id1).write(id2,buf,queue,context,size)!= CL_SUCCESS)return false;
            return true;
    }
    bool write_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,cv::Mat& buf) {
            if(_kernels.at(id1).write(id2,buf,context)!= CL_SUCCESS)return false;
            return true;
    }
    template<typename T>
    bool write_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,T&& value){
        if(_kernels.at(id1)._kernel->setArg(id2,value)!= CL_SUCCESS)return false;
        return true;
    };
    template<typename T>
    bool reload_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,T* buf) {
            if(_kernels.at(id1).reload(id2,buf,queue)!= CL_SUCCESS)return false;
            return true;
    }
    bool reload_buf(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,cv::Mat& buf) {
            if(_kernels.at(id1).reload(id2,buf,context)!= CL_SUCCESS)return false;
            return true;
    }
    cl_int run(size_t id1/*kernal ID*/,std::size_t  x=1,std::size_t y=1,std::size_t z=1) {
        cl_int err=0;
        cl::Event event;
        if(queue->flush()==CL_OUT_OF_HOST_MEMORY){
            std::cerr<<"GPU out of memory goodbye:)\n";
            exit(0);
        }
        if(z>1 && y>1){
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y,z)/*Global*/, cl::NullRange/*local*/,NULL,&event);
        }else if(z<2 && y>1){
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x,y)/*Global*/, cl::NullRange/*local*/,NULL,&event);
        }else{
            err=queue->enqueueNDRangeKernel(*_kernels.at(id1)._kernel, cl::NullRange/*offset*/, cl::NDRange(x)/*Global*/, cl::NullRange/*local*/,NULL,&event);
        };
        assert(err==CL_SUCCESS);
        event.wait();
        if(queue->finish()==CL_OUT_OF_HOST_MEMORY){
            std::cerr<<"GPU out of memory goodbye:)\n";
            exit(0);
        }
        return 1;

    }
    template<typename T>
    void read(size_t id1/*kernal ID*/,size_t id2/*buffer ID*/,size_t size/*size*/, T* out){
        assert(out);
        cl_int error;
        cl::Event event;
        T* Map_buf=(T*)queue->enqueueMapBuffer(_kernels.at(id1).read(id2).first,CL_NON_BLOCKING,CL_MAP_READ,0,sizeof(T) * size,NULL,&event,&error);
        assert(error == CL_SUCCESS);
        event.wait();
        memcpy(out,Map_buf,sizeof(T) * size);
        if(queue->enqueueUnmapMemObject(_kernels.at(id1).read(id2).first,Map_buf,NULL,&event)!=CL_SUCCESS)std::cerr<<"Read buffer on GPU failed"<<'\n';
        event.wait();
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

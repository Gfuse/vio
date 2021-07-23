// Copyright (C) 2021  Majid Geravand
// Copyright (C) 2021  Gfuse

// Enable OpenCL 32-bit integer atomic functions.
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

double2 world2cam(double f_x,double f_y,double c_x,double c_y,double s_, double x, double y,double z)
{
    double2 uv=(x/z,y/z);
    double r = sqrt(pow(uv[0],2)+pow(uv[1],2));
    double factor=1.0;
    if(s_ !=0 || r > 0.001)factor=(atan(r * 2.0 * tan(0.5 * s_)) / (r*s_));
    return double3(cx_ + fx_ * factor * uv[0],cy_ + fy_ * factor * uv[1]);
}
double3 xyz_cur(double cur_x,double cur_z,double cur_pitch,double ref_x,double ref_z,double ref_pitch,double3 ref_feature){
    if(ref_pitch<0.0)
        ref_pitch=3.141592653589793238462643383279502884197169399375-ref_pitch;
    else
        ref_pitch=ref_pitch-3.1415926535897932384626433832795028841971693993;
    double cc_ss=cos(cur_pitch)*cos(ref_pitch)-sin(cur_pitch)*sin(ref_pitch);
    double sc_cs=cos(cur_pitch)*sin(ref_pitch)+cos(ref_pitch)*sin(cur_pitch);
    return  double3(cc_ss*ref_feature.x()+sc_cs*ref_feature.z()+cur_x-ref_x*cos(cur_pitch)-ref_z*sin(cur_pitch),
                    ref_feature.y(),
                    -1.0*sc_cs*ref_feature.x())+cc_ss*ref_feature.z()+cur_z+ref_x*sin(cur_pitch)-ref_z*cos(cur_pitch));
}

__kernel void compute-residual(
    __read_only  image2d_t   image_cur, // current frame
    __read_only  image2d_t   image_ref, // reference frame
    __global     double      * cur_pose,//[reference frame pose{x,z,pitch}]
    __global     double      * ref_pose,//[current frame pose{x,z,pitch}]
    __global     double3     * ref_feature, // feature on the reference frame, when we applied the distance calculation: xyz_ref((*it)->f*((*it)->point->pos_ - Eigen::Vector3d(ref_pos[0],0.0,ref_pos[1])).norm());
    __global     int         * level,
    __global     double      * camera_model, // camera model [f_x,f_y,c_x,c_y,s_]
                 float       * errors,
                 double      * Hessian,
                 double      * Jacobian,
                 double      * chi
) {
    // Prepare a suitable OpenCL image sampler.
    sampler_t const sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    // Use global work feature.
    int  const f   = get_global_id(0);
    ref_feature[f].x;
    ref_feature[f].y;
    ref_feature[f].z;
    PATCH_SIZE;
}


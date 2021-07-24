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

double3 xyz_ref(double fts_pos_x,double fts_pos_y,double fts_pos_z,double ref_x,double ref_y,double ref_z)
{
    double x = sqrt(pow((fts_pos_x - ref_x), 2));
    double y = sqrt(pow((fts_pos_y - ref_y), 2));
    double z = sqrt(pow((fts_pos_z - ref_z), 2));
    return double3(x, y, z);
}

void jacobian_xyz2uv(double3& xyz_in_f, double6& J)
{
    double x = xyz_in_f[0];
    double y = xyz_in_f[1];
    double z_inv = 1. / xyz_in_f[2];
    double z_inv_2 = z_inv * z_inv;

    J[0] = -z_inv;              // -1/z
    J[1] = x * z_inv_2;           // x/z^2
    J[2] = -(1.0 + pow(x,2) / z_inv_2);   // -(1.0 + x^2/z^2)

    J[3] = 1e-19;                // 0
    J[4] = y * z_inv_2;           // y/z^2
    J[5] = -x * y / z_inv_2;      // -x*y/z^2
}

void compute_hessain(double3& j, double9& H, double w)
{
    H[0] += j[0] * j[0] * w;
    H[1] += j[0] * j[1] * w;
    H[2] += j[0] * j[2] * w;
    H[3] += j[1] * j[0] * w;
    H[4] += j[1] * j[1] * w;
    H[5] += j[1] * j[2] * w;
    H[6] += j[2] * j[0] * w;
    H[7] += j[2] * j[1] * w;
    H[8] += j[2] * j[2] * w;
}

void compute_Jacobian(double3& j, double& Jacobian, double r, double w)
{
    Jacobian[0] -= j[0] * r * w;
    Jacobian[1] -= j[1] * r * w;
    Jacobian[2] -= j[2] * r * w;
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
    static const int PATCH_SIZE = 4;
    static const int PATCH_HALFSIZE = 2;
    static const int PATCH_AREA_ = PATCH_SIZE * PATCH_SIZE;
    int border = PATCH_HALFSIZE + 1;
    int stride = get_image_dim(image_cur)[0]; //width, height
    double scale = pow(2, (-1.0 * level));
    double focal_length = camera_model[0];
    size_t feature_counter = 0;
    double chi2 = 0.0;
    int n_meas_ = 0;
    // Prepare a suitable OpenCL image sampler.
    sampler_t const sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    // Use global work feature.
    int const f = get_global_id(0);
    //ref_feature[f].x;
    //ref_feature[f].y;
    //ref_feature[f].z;

    double u_ref = ref_feature[f].x * scale; //??????  ref_feature[f].x
    double v_ref = ref_feature[f].y * scale; //??????  ref_feature[f].y
    int u_ref_i = floor(u_ref);
    int v_ref_i = floor(v_ref);

    // if(ref_feature[f]->point == NULL || u_ref_i - border < 0 || v_ref_i - border < 0 || //???????  ref_feature[f]->point
    if(u_ref_i - border < 0 || v_ref_i - border < 0 ||
       u_ref_i + border >= get_image_dim(image_ref)[0] || v_ref_i + border >= get_image_dim(image_ref)[1])
        return;

    xyz_reference = xyz_ref(ref_feature[f][0], ref_feature[f][1], ref_feature[f][2], ref_pose.x, ref_pose.y, ref_pose.z); // ref_pose.y ????
    // evaluate projection jacobian
    double6 frame_jac; //2X3
    jacobian_xyz2uv(xyz_reference, frame_jac);
    double subpix_u_ref = u_ref - u_ref_i;
    double subpix_v_ref = v_ref - v_ref_i;
    double w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    double w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    double w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    double w_ref_br = subpix_u_ref * subpix_v_ref;
    size_t pixel_counter = 0;
    /// compute residual ...
    double3 xyz_current = xyz_cur(cur_pose.x, cur_pose.z, cur_pose.pitch, ref_pose.x ,ref_pose.z ,ref_pose.pitch, ref_feature[f]);
    double2 uv_cur_pyr = world2cam(camera_model[0], camera_model[1], camera_model[2], camera_model[3], camera_model[4],
                                   xyz_current[0], xyz_current[1], xyz_current[2]) * scale;
    double u_cur = uv_cur_pyr[0];
    double v_cur = uv_cur_pyr[1];
    int u_cur_i = floor(u_cur);
    int v_cur_i = floor(v_cur);

    // check if projection is within the image
    if(u_cur_i < 0 || v_cur_i < 0 || u_cur_i - border < 0 || v_cur_i - border < 0 ||
    u_cur_i + border >= get_image_dim(image_cur)[0] || v_cur_i + border >= get_image_dim(image_cur)[0])
    return;
    // compute bilateral interpolation weights for the current image
    double subpix_u_cur = u_cur - u_cur_i;
    double subpix_v_cur = v_cur - v_cur_i;
    double w_cur_tl = (1.0 - subpix_u_cur) * (1.0 - subpix_v_cur);
    double w_cur_tr = subpix_u_cur * (1.0 - subpix_v_cur);
    double w_cur_bl = (1.0 - subpix_u_cur) * subpix_v_cur;
    double w_cur_br = subpix_u_cur * subpix_v_cur;

    /// TODO - check & implementation
    // float* ref_patch_cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
    /// ... compute residual

    // float* cache_ptr = reinterpret_cast<float*>(ref_patch_cache_.data) + patch_area_*feature_counter;
    for(int y = 0; y < PATCH_SIZE; ++y)
    {
        /// TODO - check & implementation
        // uint8_t* ref_img_ptr = (uint8_t*) ref_img.data + (v_ref_i+y-patch_halfsize_)*stride + (u_ref_i-patch_halfsize_);
        // uint8_t* cur_img_ptr = (uint8_t*) image_cur.data + (v_cur_i + y - patch_halfsize_) * stride + (u_cur_i - patch_halfsize_);

        // for(int x = 0; x < PATCH_SIZE; ++x, ++pixel_counter, ++ref_img_ptr, ++cache_ptr)
        // for(int x = 0; x < PATCH_SIZE; ++x, ++pixel_counter, ++cur_img_ptr, ++ref_patch_cache_ptr)
        for(int x = 0; x < PATCH_SIZE; ++x, ++pixel_counter, ++ref_img_ptr, ++cur_img_ptr)
        {
            /// TODO - check & implementation
            // precompute interpolated reference patch color
            // *cache_ptr = w_ref_tl*ref_img_ptr[0] + w_ref_tr*ref_img_ptr[1] + w_ref_bl*ref_img_ptr[stride] + w_ref_br*ref_img_ptr[stride+1];

            // we use the inverse compositional: thereby we can take the gradient always at the same position
            // get gradient of warped image (~gradient at warped position)
            double dx = 0.5f * ((w_ref_tl * ref_img_ptr[1] + w_ref_tr * ref_img_ptr[2] + w_ref_bl * ref_img_ptr[stride+1] + w_ref_br * ref_img_ptr[stride+2])
                               -(w_ref_tl * ref_img_ptr[-1] + w_ref_tr * ref_img_ptr[0] + w_ref_bl * ref_img_ptr[stride-1] + w_ref_br * ref_img_ptr[stride]));
            double dy = 0.5f * ((w_ref_tl * ref_img_ptr[stride] + w_ref_tr * ref_img_ptr[1+stride] + w_ref_bl * ref_img_ptr[stride*2] + w_ref_br * ref_img_ptr[stride*2+1])
                               -(w_ref_tl * ref_img_ptr[-stride] + w_ref_tr * ref_img_ptr[1-stride] + w_ref_bl * ref_img_ptr[0] + w_ref_br * ref_img_ptr[1]));

            // compute residual
            /// TODO - check & implementation
            // double intensity_cur = w_cur_tl * cur_img_ptr[0] + w_cur_tr * cur_img_ptr[1] + w_cur_bl * cur_img_ptr[stride] + w_cur_br * cur_img_ptr[stride+1];
            double res = intensity_cur - (ref_patch_cache_ptr[x]);

            // used to compute scale for robust cost
            if(compute_weight_scale)
            errors[(y * PATCH_SIZE) + x] = fabs(res)

            // robustification
            double weight = 2.0;
            // if(use_weights_) {  // use_weights_ dafault value is false
            //     weight = weight_function_->value(res/scale_);
            // }

            chi2 += res * res * weight;
            n_meas_++;

            // if(linearize_system)
            // {
            // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
            double3 j_row0 = double3(dx * frame_jac[0], dx * frame_jac[1], dx * frame_jac[2])
            double3 j_row1 = double3(dy * frame_jac[3], dy * frame_jac[4], dy * frame_jac[5])
            double3 J(j_row0 + j_row1) * (focal_length / scale);
            compute_hessain(j, Hessian, weight); // H_.noalias() += J * J.transpose() * weight;
            compute_Jacobian(j, Jacobian, res, weight) // Jres_.noalias() -= J * res * weight;
            // }
            chi = chi2 / n_meas_;
            feature_counter++;
        }
    }
}


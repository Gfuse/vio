// Copyright (C) 2021  Majid Geravand
// Copyright (C) 2021  Gfuse

// Enable OpenCL 32-bit integer atomic functions.
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

double2 world2cam(double f_x, double f_y, double c_x, double c_y, double s_, double x, double y, double z)
{
    double2 uv = (x/z, y/z);
    double r = sqrt(pow(uv[0], 2) + pow(uv[1], 2));
    double factor = 1.0;
    if(s_ != 0 || r > 0.001)
        factor = (atan(r * 2.0 * tan(0.5 * s_)) / (r * s_));
    return double3(cx_ + fx_ * factor * uv[0], cy_ + fy_ * factor * uv[1]);
}

double3 xyz_cur(double cur_x, double cur_z, double cur_pitch, double ref_x, double ref_z, double ref_pitch, double3 ref_feature)
{
    if(ref_pitch < 0.0)
        ref_pitch = 3.141592653589793238462643383279502884197169399375 - ref_pitch;
    else
        ref_pitch = ref_pitch - 3.1415926535897932384626433832795028841971693993;
    double cc_ss = cos(cur_pitch) * cos(ref_pitch) - sin(cur_pitch) * sin(ref_pitch);
    double sc_cs = cos(cur_pitch) * sin(ref_pitch) + cos(ref_pitch) * sin(cur_pitch);
    return  double3(cc_ss * ref_feature.x() + sc_cs * ref_feature.z() + cur_x - ref_x * cos(cur_pitch) - ref_z * sin(cur_pitch),
                    ref_feature.y(),
                    -1.0 * sc_cs * ref_feature.x() + cc_ss * ref_feature.z() + cur_z + ref_x * sin(cur_pitch) - ref_z * cos(cur_pitch));
}

double3 xyz_ref(double fts_pos_x, double fts_pos_y, double fts_pos_z, double ref_x, double ref_y, double ref_z)
{
    return double3(sqrt(pow((fts_pos_x - ref_x), 2)),
                   sqrt(pow((fts_pos_y - ref_y), 2)),
                   sqrt(pow((fts_pos_z - ref_z), 2)));
}

void jacobian_xyz2uv(double3 xyz_in_f, double* J)
{
    *(J + 0) = -(1. / xyz_in_f[2]);                                                         // -1/z
    *(J + 1) = (xyz_in_f[0]) * ((1. / xyz_in_f[2]) * (1. / xyz_in_f[2]));                   // x/z^2
    *(J + 2) = -(1.0 + pow((xyz_in_f[0]),2) / ((1. / xyz_in_f[2]) * (1. / xyz_in_f[2])));   // -(1.0 + x^2/z^2)
    *(J + 3) = 1e-19;                                                                       // 0
    *(J + 4) = (xyz_in_f[1]) * ((1. / xyz_in_f[2]) * (1. / xyz_in_f[2]));                   // y/z^2
    *(J + 5) = -(xyz_in_f[0]) * (xyz_in_f[1]) / ((1. / xyz_in_f[2]) * (1. / xyz_in_f[2]));  // -x*y/z^2
}

void compute_hessain(double3 j, double* H, double w)
{
    *(H + 0) += j[0] * j[0] * w;
    *(H + 1) += j[0] * j[1] * w;
    *(H + 2) += j[0] * j[2] * w;
    *(H + 3) += j[1] * j[0] * w;
    *(H + 4) += j[1] * j[1] * w;
    *(H + 5) += j[1] * j[2] * w;
    *(H + 6) += j[2] * j[0] * w;
    *(H + 7) += j[2] * j[1] * w;
    *(H + 8) += j[2] * j[2] * w;
}

void compute_Jacobian(double3 j, double* Jacobian, double r, double w)
{
    *(Jacobian + 0) -= j[0] * r * w;
    *(Jacobian + 1) -= j[1] * r * w;
    *(Jacobian + 2) -= j[2] * r * w;
}

__kernel void compute-residual(
    __read_only  image2d_t   image_cur, // current frame
    __read_only  image2d_t   image_ref, // reference frame
    __global     double      * cur_pose,//[reference frame pose{x,z,pitch}]
    __global     double      * ref_pose,//[current frame pose{x,z,pitch}]
    __global     double3     * ref_feature, // feature on the reference frame, when we applied the distance calculation: xyz_ref((*it)->f*((*it)->point->pos_ - Eigen::Vector3d(ref_pos[0],0.0,ref_pos[1])).norm());
    __global     double2     * featue_px,
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
    size_t n_meas_ = 0;
    // Prepare a suitable OpenCL image sampler.
    sampler_t const sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    // Use global work feature.
    int const f = get_global_id(0);

    // check if reference with patch size is within image
    double u_ref = featue_px[f].x * scale;
    double v_ref = featue_px[f].y * scale;
    int u_ref_i = floor(u_ref);
    int v_ref_i = floor(v_ref);

    if(u_ref_i - (PATCH_HALFSIZE + 1) < 0 || v_ref_i - (PATCH_HALFSIZE + 1) < 0 ||
       u_ref_i + (PATCH_HALFSIZE + 1) >= get_image_dim(image_ref)[0] || v_ref_i + (PATCH_HALFSIZE + 1) >= get_image_dim(image_ref)[1])
        return;

    xyz_reference = xyz_ref(ref_feature[f][0], ref_feature[f][1], ref_feature[f][2], ref_pose.x, ref_pose.y, ref_pose.z);

    // evaluate projection jacobian
    double* frame_jac = (double *) malloc(6 * sizeof(double)); // 2X3
    jacobian_xyz2uv(xyz_reference, frame_jac);

    // compute bilateral interpolation weights for reference image
    double subpix_u_ref = u_ref - u_ref_i;
    double subpix_v_ref = v_ref - v_ref_i;
    double w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    double w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    double w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    double w_ref_br = subpix_u_ref * subpix_v_ref;
    size_t pixel_counter = 0;

    double3 xyz_current = xyz_cur(cur_pose.x, cur_pose.z, cur_pose.pitch, ref_pose.x ,ref_pose.z ,ref_pose.pitch, ref_feature[f]);
    double2 uv_cur_pyr = world2cam(camera_model[0], camera_model[1], camera_model[2], camera_model[3], camera_model[4],
                                   xyz_current[0], xyz_current[1], xyz_current[2]) * scale;
    double u_cur = uv_cur_pyr[0];
    double v_cur = uv_cur_pyr[1];
    int u_cur_i = floor(u_cur);
    int v_cur_i = floor(v_cur);

    // check if projection is within the image
    if(u_cur_i < 0 || v_cur_i < 0 || u_cur_i - (PATCH_HALFSIZE + 1) < 0 || v_cur_i - (PATCH_HALFSIZE + 1) < 0 ||
       u_cur_i + (PATCH_HALFSIZE + 1) >= get_image_dim(image_cur)[0] || v_cur_i + (PATCH_HALFSIZE + 1) >= get_image_dim(image_cur)[1])
        return;

    // compute bilateral interpolation weights for the current image
    double subpix_u_cur = u_cur - u_cur_i;
    double subpix_v_cur = v_cur - v_cur_i;
    double w_cur_tl = (1.0 - subpix_u_cur) * (1.0 - subpix_v_cur);
    double w_cur_tr = subpix_u_cur * (1.0 - subpix_v_cur);
    double w_cur_bl = (1.0 - subpix_u_cur) * subpix_v_cur;
    double w_cur_br = subpix_u_cur * subpix_v_cur;

    for(int y = 0; y < PATCH_SIZE; ++y)
    {
        int ref_element_addr = (v_ref_i + y - PATCH_HALFSIZE) * get_image_dim(image_ref)[0] + (u_ref_i - PATCH_HALFSIZE);
        int cur_element_addr = (v_cur_i + y - PATCH_HALFSIZE) * get_image_dim(image_cur)[0] + (u_cur_i - PATCH_HALFSIZE);

        for(int x = 0; x < PATCH_SIZE; ++x, ++pixel_counter, ++ref_element_addr, ++cur_element_addr)
        {
            // precompute interpolated reference patch color
            int2 px_reftl = ( ref_element_addr                                    % get_image_dim(image_ref)[0], ref_element_addr                                     / get_image_dim(image_ref)[0]);
            int2 px_reftr = ((ref_element_addr + 1)                               % get_image_dim(image_ref)[0], (ref_element_addr + 1)                               / get_image_dim(image_ref)[0]);
            int2 px_refbl = ((ref_element_addr + get_image_dim(image_ref)[0])     % get_image_dim(image_ref)[0], (ref_element_addr + get_image_dim(image_ref)[0])     / get_image_dim(image_ref)[0]);
            int2 px_refbr = ((ref_element_addr + get_image_dim(image_ref)[0] + 1) % get_image_dim(image_ref)[0], (ref_element_addr + get_image_dim(image_ref)[0] + 1) / get_image_dim(image_ref)[0]);
            double value = w_ref_tl * read_imageui(image_ref, sampler, px_reftl).x + w_ref_tr * read_imageui(image_ref, sampler, px_reftr).x +
                           w_ref_bl * read_imageui(image_ref, sampler, px_refbl).x + w_ref_br * read_imageui(image_ref, sampler, px_refbr).x;

            double dx = 0.5f * ((w_ref_tl * read_imageui(image_ref, sampler, ((ref_element_addr + 1)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 1)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, ((ref_element_addr + 2)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 2)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]) + 2)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) + 2)     / get_image_dim(image_ref)[0]))).x)
                               -(w_ref_tl * read_imageui(image_ref, sampler, ((ref_element_addr - 1)                                     % get_image_dim(image_ref)[0], ((ref_element_addr - 1)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, ((ref_element_addr + 0)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 0)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]) - 1)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) - 1)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]))         % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]))         / get_image_dim(image_ref)[0]))).x));

            double dy = 0.5f * ((w_ref_tl * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]))         % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]))         / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2 + 1) % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2 + 1) / get_image_dim(image_ref)[0]))).x)
                               -(w_ref_tl * read_imageui(image_ref, sampler, ((ref_element_addr + (-get_image_dim(image_ref)[0]))        % get_image_dim(image_ref)[0], ((ref_element_addr + (-get_image_dim(image_ref)[0]))        / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, ((ref_element_addr + (1 - get_image_dim(image_ref)[0]))     % get_image_dim(image_ref)[0], ((ref_element_addr + (1 - get_image_dim(image_ref)[0]))     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, ((ref_element_addr + 0)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 0)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, ((ref_element_addr + 1)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 1)                                     / get_image_dim(image_ref)[0]))).x));

            // compute residual
            int2 px_curtl = ( cur_element_addr                                      % get_image_dim(image_cur)[0], cur_element_add                                       / get_image_dim(image_cur)[0]);
            int2 px_curtr = ((cur_element_addr + 1)                                 % get_image_dim(image_cur)[0], (cur_element_add + 1)                                 / get_image_dim(image_cur)[0]);
            int2 px_curbl = ((cur_element_addr + (get_image_dim(image_cur)[0]))     % get_image_dim(image_cur)[0], (cur_element_add + (get_image_dim(image_cur)[0]))     / get_image_dim(image_cur)[0]);
            int2 px_curbr = ((cur_element_addr + (get_image_dim(image_cur)[0]) + 1) % get_image_dim(image_cur)[0], (cur_element_add + (get_image_dim(image_cur)[0]) + 1) / get_image_dim(image_cur)[0]);
            double intensity_cur = w_cur_tl * read_imageui(image_cur, sampler, px_curtl).x + w_cur_tr * read_imageui(image_cur, sampler, px_curtr).x +
                                   w_cur_bl * read_imageui(image_cur, sampler, px_curbl).x + w_cur_br * read_imageui(image_cur, sampler, px_curbr).x;

            double res = intensity_cur - value;

            // used to compute scale for robust cost
            // if(compute_weight_scale)
                errors[f * pow(PATCH_SIZE, 2) + (y * PATCH_SIZE) + x] = fabs(res);

            // robustification
            double weight = 2.0;
            // if(use_weights_) {  // use_weights_ dafault value is false
            //     weight = weight_function_->value(res/scale_);
            // }

            *chi += res * res * weight;
            n_meas_++;

            // if(linearize_system)
            // {
                // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
                double3 j_row0 = double3(dx * frame_jac[0], dx * frame_jac[1], dx * frame_jac[2])
                double3 j_row1 = double3(dy * frame_jac[3], dy * frame_jac[4], dy * frame_jac[5])
                double3 J((j_row0 + j_row1) * (camera_model[0] / scale));
                compute_hessain(J, Hessian, weight); // H_.noalias() += J * J.transpose() * weight;
                compute_Jacobian(J, Jacobian, res, weight) // Jres_.noalias() -= J * res * weight;
            // }
        }
    }
    *chi = *chi / n_meas_;
    free(frame_jac);
}


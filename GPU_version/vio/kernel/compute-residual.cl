// Copyright (C) 2021  Majid Geravand
// Copyright (C) 2021  Gfuse

// Enable OpenCL 32-bit integer atomic functions.
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

double2 world2cam(double3 feature)
{
    double r = sqrt(pow(feature.x/feature.z, 2) + pow(feature.y/feature.z, 2));
    double factor = 1.0;
    if(S != 0 || r > 0.001)
        factor = (atan(r * 2.0 * tan(0.5 * S)) / (r * S));
    return (double2)(C_X + F_X * factor * feature.x/feature.z, C_Y + F_Y * factor * feature.y/feature.z);
}

double3 xyz_cur(double3 cur, double3 ref, double3 ref_feature)
{
    if(ref.z < 0.0)
        ref.z = 3.141592653589793238462643383279502884197169399375 - ref.z;
    else
        ref.z = ref.z - 3.1415926535897932384626433832795028841971693993;
    double cc_ss = cos(cur.z) * cos(ref.z) - sin(cur.z) * sin(ref.z);
    double sc_cs = cos(cur.z) * sin(ref.z) + cos(ref.z) * sin(cur.z);
    return  (double3)(cc_ss * ref_feature.x + sc_cs * ref_feature.z + cur.x - ref.x * cos(cur.z) - ref.z * sin(cur.z),
                      ref_feature.y,
                      -1.0 * sc_cs * ref_feature.x + cc_ss * ref_feature.z + cur.z + ref.x * sin(cur.z) - ref.z * cos(cur.z));
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
void compute_hessain(double3 j, __global double* H, double w)
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

__kernel void compute_residual(
        __read_only  image2d_t   image_cur, // current frame
        __read_only  image2d_t   image_ref, // reference frame
        __global     double3      * cur_pose,//[reference frame pose{x,z,pitch}]
        __global     double3      * ref_pose,//[current frame pose{x,z,pitch}]
        __global     double3     * ref_feature, // feature on the reference frame, when we applied the distance calculation: xyz_ref((*it)->f*((*it)->point->pos_ - Eigen::Vector3d(ref_pos[0],0.0,ref_pos[1])).norm());
        __global     double2     * featue_px,
                        int        level,
        __global     float       * errors,
        __global     double      * Hessian,
        __global     double3      * Jacobian,
        __global     double      * chi,
                     double        scale_
)
{
    double scale = pow(2.0, -level);
    // Prepare a suitable OpenCL image sampler.
    sampler_t const sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    // Use global work feature.
    int f = get_global_id(0);
    // check if reference with patch size is within image
    double2 uv_ref = featue_px[f] * scale;
    double2 uv_ref_i = floor(uv_ref);
    if(uv_ref_i.x - (PATCH_HALFSIZE + 1) < 0 || uv_ref_i.y - (PATCH_HALFSIZE + 1) < 0 || uv_ref_i.x + (PATCH_HALFSIZE + 1) >= get_image_dim(image_ref)[0] || uv_ref_i.y + (PATCH_HALFSIZE + 1) >= get_image_dim(image_ref)[1])return;
    // evaluate projection jacobian
    double frame_jac[6]={0.0}; // 2X3
    jacobian_xyz2uv(sqrt(pow(ref_feature[f] - (double3)(ref_pose[0].x,0.0,ref_pose[0].y), 2.0)), &frame_jac);

    // compute bilateral interpolation weights for reference image
    double2 subpix=  uv_ref - uv_ref_i;
    double w_ref_tl = (1.0 - subpix.x) * (1.0 - subpix.y);
    double w_ref_tr = subpix.x * (1.0 - subpix.y);
    double w_ref_bl = (1.0 - subpix.x) * subpix.y;
    double w_ref_br = subpix.x * subpix.y;

    // compute bilateral interpolation weights for the current image
    double2 uv_cur_pyr = world2cam(xyz_cur(cur_pose[0], ref_pose[0] ,ref_feature[f])) * scale;
    double2 uv_cur_i = floor(uv_cur_pyr);
    double2 subpix_uv_cur = uv_cur_pyr - uv_cur_i;
    double w_cur_tl = (1.0 - subpix_uv_cur.x) * (1.0 - subpix_uv_cur.y);
    double w_cur_tr = subpix_uv_cur.x * (1.0 - subpix_uv_cur.y);
    double w_cur_bl = (1.0 - subpix_uv_cur.x) * subpix_uv_cur.y;
    double w_cur_br = subpix_uv_cur.x * subpix_uv_cur.y;
    float e = 0.0;
    double chi_=0.0;
    for(int y = 0; y < PATCH_SIZE; ++y)
    {
        int ref_element_addr = (int)(uv_ref_i.y + y - PATCH_HALFSIZE) * get_image_dim(image_ref)[0] + (uv_ref_i.x - PATCH_HALFSIZE);
        int cur_element_addr = (int)(uv_cur_i.y + y - PATCH_HALFSIZE) * get_image_dim(image_cur)[0] + (uv_cur_i.x - PATCH_HALFSIZE);

        for(int x = 0; x < PATCH_SIZE; ++x, ++ref_element_addr, ++cur_element_addr)
        {
            // precompute interpolated reference patch color
            int2 px_reftl = (int2)( ref_element_addr                                    % get_image_dim(image_ref)[0],  ref_element_addr                                    / get_image_dim(image_ref)[0]);
            int2 px_reftr = (int2)((ref_element_addr + 1)                               % get_image_dim(image_ref)[0], (ref_element_addr + 1)                               / get_image_dim(image_ref)[0]);
            int2 px_refbl = (int2)((ref_element_addr + get_image_dim(image_ref)[0])     % get_image_dim(image_ref)[0], (ref_element_addr + get_image_dim(image_ref)[0])     / get_image_dim(image_ref)[0]);
            int2 px_refbr = (int2)((ref_element_addr + get_image_dim(image_ref)[0] + 1) % get_image_dim(image_ref)[0], (ref_element_addr + get_image_dim(image_ref)[0] + 1) / get_image_dim(image_ref)[0]);
            double value = w_ref_tl * read_imageui(image_ref, sampler, px_reftl).x + w_ref_tr * read_imageui(image_ref, sampler, px_reftr).x +
                           w_ref_bl * read_imageui(image_ref, sampler, px_refbl).x + w_ref_br * read_imageui(image_ref, sampler, px_refbr).x;

            double dx = 0.5f * ((w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 1)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 1)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 2)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 2)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]) + 2)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) + 2)     / get_image_dim(image_ref)[0]))).x)
                               -(w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr - 1)                                     % get_image_dim(image_ref)[0], ((ref_element_addr - 1)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 0)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 0)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]) - 1)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) - 1)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]))         % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]))         / get_image_dim(image_ref)[0]))).x));

            double dy = 0.5f * ((w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]))         % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]))         / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) + 1)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2)     % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2)     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2 + 1) % get_image_dim(image_ref)[0], ((ref_element_addr + (get_image_dim(image_ref)[0]) * 2 + 1) / get_image_dim(image_ref)[0]))).x)
                               -(w_ref_tl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (-get_image_dim(image_ref)[0]))        % get_image_dim(image_ref)[0], ((ref_element_addr + (-get_image_dim(image_ref)[0]))        / get_image_dim(image_ref)[0]))).x +
                                 w_ref_tr * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + (1 - get_image_dim(image_ref)[0]))     % get_image_dim(image_ref)[0], ((ref_element_addr + (1 - get_image_dim(image_ref)[0]))     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_bl * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 0)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 0)                                     / get_image_dim(image_ref)[0]))).x +
                                 w_ref_br * read_imageui(image_ref, sampler, (int2) ((ref_element_addr + 1)                                     % get_image_dim(image_ref)[0], ((ref_element_addr + 1)                                     / get_image_dim(image_ref)[0]))).x));

            // compute residual
            int2 px_curtl = (int2)( cur_element_addr                                      % get_image_dim(image_cur)[0],  cur_element_addr                                      / get_image_dim(image_cur)[0]);
            int2 px_curtr = (int2)((cur_element_addr + 1)                                 % get_image_dim(image_cur)[0], (cur_element_addr + 1)                                 / get_image_dim(image_cur)[0]);
            int2 px_curbl = (int2)((cur_element_addr + (get_image_dim(image_cur)[0]))     % get_image_dim(image_cur)[0], (cur_element_addr + (get_image_dim(image_cur)[0]))     / get_image_dim(image_cur)[0]);
            int2 px_curbr = (int2)((cur_element_addr + (get_image_dim(image_cur)[0]) + 1) % get_image_dim(image_cur)[0], (cur_element_addr + (get_image_dim(image_cur)[0]) + 1) / get_image_dim(image_cur)[0]);

            double res = value - w_cur_tl * read_imageui(image_cur, sampler, px_curtl).x + w_cur_tr * read_imageui(image_cur, sampler, px_curtr).x +
                                 w_cur_bl * read_imageui(image_cur, sampler, px_curbl).x + w_cur_br * read_imageui(image_cur, sampler, px_curbr).x;
            // used to compute scale for robust cost
            e += fabs(res);
            // robustification
            double weight = res/scale_; //1.48f * vk::getMedian(errors)
            chi_ += pow(res,2) * weight;
            double3 j_row0 = (double3)((dx * frame_jac[0]), (dx * frame_jac[1]), (dx * frame_jac[2]));
            double3 j_row1 = (double3)((dy * frame_jac[3]), (dy * frame_jac[4]), (dy * frame_jac[5]));
            double3 J = (j_row0 + j_row1) * (F_X / scale);
            compute_hessain(J, &Hessian[f*9], weight);
            Jacobian[f] -= J * res * weight;

        }
    }
    errors[f] = e / pow(PATCH_SIZE,2.0);
    chi[f]=chi_;
}


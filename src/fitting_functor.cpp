#include "fitting_functor.h"

#include <sstream>
#include <filesystem>
#include <iomanip>

using namespace Eigen;

int FittingFunctor::operator()(const InputType &x, ValueType &fvec)
{
    evaluator.evaluate_subdiv_surface(mesh.vertices(), x.correspondences, sf, false);
    // fill residuals
    for (int i = 0; i < points->size(); ++i)
    {
        fvec.segment(i * 3, 3) = sf.S[i] - points->at(i);
    }
    
    return 0;
}

int FittingFunctor::df(const InputType &x, JacobianType &fjac)
{
    evaluator.evaluate_subdiv_surface(mesh.vertices(), x.correspondences, sf);
    const int n_points = points->size();

    // derivative wrt vertices
    SparseMatrix<double> dS_dV(n_points, mesh.n_vertices());
    dS_dV.setFromTriplets(sf.dSdX.begin(), sf.dSdX.end());

    // compute non-zero entries in jacobian
    TripletArray<double, JacobianType::Index> jvals(
        3 * n_points // for each residual
        * (2         // u,v
           + 1       // scale along corresponding coordinate
           + 1       // translation along corresponding coordinate
           ));

    for (int i = 0; i < n_points; ++i)
    {
        for (int j{0}; j < 3; ++j) // x, y, z for each point
        {
            // derivatives wrt correspondences
            jvals.add(3 * i + j, 2 * i + 0, sf.Su[i](j)); // dfi_j / du
            jvals.add(3 * i + j, 2 * i + 1, sf.Sv[i](j)); // dfi_j / dv

            // derivatives wrt parameters
            for (int k{0}; k < mesh.n_vertices(); ++k)
            {
                // +j in dS_dV is skipped in below expression because it is the same for all j
                jvals.add(3 * i + j, 2 * n_points + 3 * 0 + j, dS_dV.coeff(i, k) * template_mesh.vertices()[k](j)); // dfi_j / ds_j
                jvals.add(3 * i + j, 2 * n_points + 3 * 1 + j, dS_dV.coeff(i, k) * 1);                              // dfi_j / dt_j
            }
        }
    }

    fjac.resize(
        3 * n_points, // n_residuals
        2 * n_points  // n_correspondences (u+v)
            + 3       // scale
            + 3       // translation
    );
    fjac.setFromTriplets(jvals.begin(), jvals.end());
    fjac.makeCompressed();

    return 0;
}

void FittingFunctor::increment_in_place(InputType *x, const StepType &p)
{
    int n_points = x->correspondences.size();
    int id_param_start = n_points * 2;

    // increment parameters
    x->scale += p.segment<3>(id_param_start);
    x->translation += p.segment<3>(id_param_start + 3);
    
    // increment correspondences
    int loopers = 0;
    int total_hops = 0;
    for (int i{0}; i < n_points; ++i)
    {
        Vector2d du = p.segment<2>(2 * i);
        int n_hops = update_surface_point(mesh, evaluator, x->correspondences[i].face, x->correspondences[i].u, du);
        if (n_hops < 0)
            ++loopers;
        total_hops += std::abs(n_hops);
    }

    // if (loopers > 0)
    //     std::cerr << "[" << total_hops << "/" << static_cast<double>(n_points) << " hops, " << loopers << " points looped]";
    // else if (total_hops > 0)
    //     std::cerr << "[" << total_hops << "/" << static_cast<double>(n_points) << " hops]";

    // update mesh in vis
    mesh.vertices() = template_mesh.vertices();  // resets to template
    mesh.transform(x->scale, x->translation);
    vis_ptr->update(mesh.tri_mesh);
    vis_ptr->show();
    iteration += 1;

    if (!dir_path.empty())
    {
        std::stringstream ss;
        ss << dir_path.string() << "/" << std::setw(2) << std::setfill('0') << iteration << ".png";
        vis_ptr->save_image(ss.str());
    }
    
    // std::cout << "Iteration: " << iteration << "\n";
}

double FittingFunctor::estimateNorm(const InputType &x, const StepType &diag)
{
    /* scale norm(InputType) by diag */

    // scale parameters
    Map<VectorXd> diag_params(const_cast<double *>(diag.data()) + x.correspondences.size() * 2, 6);
    // double total = (diag_params.segment<1>(0) * x.radius).squaredNorm();
    double total{0.0};
    total += diag_params.segment<3>(0).cwiseProduct(x.scale).squaredNorm();
    total += diag_params.segment<3>(3).cwiseProduct(x.translation).squaredNorm();

    // scale correspondences
    for (int i{0}; i < x.correspondences.size(); ++i)
    {
        const Vector2d &u = x.correspondences[i].u;
        Vector2d di = diag.segment<2>(2 * i); // correspoding diagonal elements
        total += u.cwiseProduct(di).squaredNorm();
    }

    return sqrt(total);
}

int FittingFunctor::update_surface_point(const Mesh &mesh, SubdivEvaluator &evaluator, int &face, Vector2d &u, const Vector2d &du)
{
    constexpr int max_hops = 7;
    int face_old = face;
    double u1_old = u(0);
    double u2_old = u(1);
    double du1 = du(0);
    double du2 = du(1);
    double u1_new = u1_old + du1;
    double u2_new = u2_old + du2;

    for (int count{0};; ++count)
    {
        bool crossing = (u1_new < 0.0) || (u2_new < 0.0) || ((u1_new + u2_new) > 1.0);

        if (!crossing)
        {
            face = face_old;
            u << u1_new, u2_new;
            return count;
        }

        // find the new face and coordinates of the crossing point within the old face and the new face
        int face_new;
        bool face_found = false;
        double u1_cross, u2_cross, aux;
        /* equation of line concept:
        * v_new = v_old + dv/du * (u_cross - u_old)
        * aux is used to compute offset in new face
        */
        if (u1_new < 0.0)
        {
            u1_cross = 0.0;
            u2_cross = u2_old + (du2 / du1) * (u1_cross - u1_old);
            aux = u2_cross;
            if ((u2_cross >= 0.0) && (u2_cross <= 1.0))
            {
                face_new = mesh.face_adj(2, face_old);
                face_found = true;
            }
        }
        if ((u2_new < 0.0) && (!face_found))
        {
            u2_cross = 0;
            u1_cross = u1_old + (du1 / du2) * (u2_cross - u2_old);
            aux = u1_cross;
            if ((u1_cross >= 0.0) && (u1_cross <= 1.0))
            {
                face_new = mesh.face_adj(0, face_old);
                face_found = true;
            }
        }
        if (((u1_new + u2_new) > 1.0) && (!face_found))
        {
            const double m = du2 / du1;
            u1_cross = (1 - u2_old + m * u1_old) / (m + 1);
            u2_cross = 1 - u1_cross;
            aux = u1_cross;
            if ((u1_cross >= 0.0) && (u1_cross <= 1.0))
            {
                face_new = mesh.face_adj(1, face_old);
                face_found = true;
            }
        }
        assert(face_found);

        // find the coordinates of the crossing point as part of the new face
        // and update u_old (this will be new u in the next iter)
        int face_adj_of_old_wrt_new;
        for (int f{0}; f < 3; ++f)
        {
            if (mesh.face_adj(f, face_new) == face_old)
            {
                face_adj_of_old_wrt_new = f;
                break;
            }
        }
        switch (face_adj_of_old_wrt_new)
        {
        case 0:
            u1_old = aux;
            u2_old = 0.0;
            break;
        case 1:
            u1_old = aux;
            u2_old = 1.0 - aux;
            break;
        case 2:
            u1_old = 0.0;
            u2_old = aux;
            break;
        }

        // evaluate subdiv surface at edge (wrt original face)
        std::vector<SurfacePoint> pts;
        pts.push_back({face, {u1_cross, u2_cross}}); // crossing point wrt face
        pts.push_back({face_new, {u1_old, u2_old}}); // crossing point wrt face new
        SurfaceFeatures sf1(pts.size());
        evaluator.evaluate_subdiv_surface(mesh.vertices(), pts, sf1);

        Eigen::Matrix<double, 3, 2> J_Sa;
        J_Sa.col(0) = sf1.Su[0];
        J_Sa.col(1) = sf1.Sv[0];

        Eigen::Matrix<double, 3, 2> J_Sb;
        J_Sb.col(0) = sf1.Su[1];
        J_Sb.col(1) = sf1.Sv[1];

        // compute new u increments
        Vector2d du_remaining;
        du_remaining << u1_new - u1_cross, u2_new - u2_cross;
        Vector3d prod = J_Sa * du_remaining;
        Eigen::Matrix2d AtA = J_Sb.transpose() * J_Sb;
        Vector2d AtB = J_Sb.transpose() * prod;
        Vector2d u_incr = AtA.inverse() * AtB;
        du1 = u_incr[0];
        du2 = u_incr[1];

        if (count == max_hops)
        {
            double dmax = std::max(du1, du2);
            double scale = 0.5 / dmax;
            face = face_old;
            u << 0.3, 0.3;
            return -count;
        }

        // update for next iter
        u1_new = u1_old + du1;
        u2_new = u2_old + du2;
        face_old = face_new;
    }
}

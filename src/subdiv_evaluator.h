#pragma once

#include "mesh.h"

#include <Eigen/Eigen>

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>

#include <Open3D/Open3D.h>

#include <vector>

struct OSDVertex
{
    OSDVertex() {}

    void Clear(void * = 0)
    {
        point.setZero();
    }

    void AddWithWeight(const OSDVertex &src, float weight)
    {
        point += weight * src.point;
    }

    void SetPosition(float x, float y, float z)
    {
        point << x, y, z;
    }

    Eigen::Vector3d point;
};

struct SurfacePoint
{
    int face;
    Eigen::Vector2d u;

    static std::vector<SurfacePoint> generate(const int n_points, const int n_faces);
};

struct SurfaceFeatures
{
    using triplets_t = Eigen::TripletArray<double>;

    static constexpr int g_max_n_weights{16};  // 16 since using ENDCAP_BSPLINE_BASIS

    std::vector<Eigen::Vector3d> S;
    std::vector<Eigen::Vector3d> Su;
    std::vector<Eigen::Vector3d> Sv;
    
    std::vector<Eigen::Vector3d> N;
 
    triplets_t dSdX;
    triplets_t dSudX;
    triplets_t dSvdX;

    float p_weights[g_max_n_weights];   // position
    float du_weights[g_max_n_weights];  // derivative wrt u
    float dv_weights[g_max_n_weights];  // derivative wrt v

    SurfaceFeatures(int n_surface_points)
    {
        S.resize(n_surface_points);
        Su.resize(n_surface_points);
        Sv.resize(n_surface_points);

        N.resize(n_surface_points);
 
        dSdX.reserve(g_max_n_weights * n_surface_points);
        dSudX.reserve(g_max_n_weights * n_surface_points);
        dSvdX.reserve(g_max_n_weights * n_surface_points);
    }

    void set_zero()
    {
        zero_vec(S);
        zero_vec(Su);
        zero_vec(Sv);
        
        zero_vec(N);

        dSdX.resize(0);
        dSudX.resize(0);
        dSvdX.resize(0);
    }

    void update(Eigen::Vector3d& vert, int idx_pt, int idx_cv)
    {
        S[idx_pt] += vert * p_weights[idx_cv];
        Su[idx_pt] += vert * du_weights[idx_cv];
        Sv[idx_pt] += vert * dv_weights[idx_cv];
    }

    void compute_normal()
    {
        for (int i{0}; i < Su.size(); ++i)
        {
            N[i] = Su[i].cross(Sv[i]);
        }
    }

private:
    static void zero_vec(std::vector<Eigen::Vector3d> &vec)
    {
        for (auto &elem : vec)
            elem.setZero();
    }
};

struct SubdivEvaluator
{
    int n_vertices;
    int n_refiner_vertices;

    mutable std::vector<OSDVertex> evaluation_verts_buffer;
    static const int maxlevel{3};
    OpenSubdiv::Far::TopologyRefiner *refiner;

    size_t n_local_points;
    OpenSubdiv::Far::PatchTable *patch_table;

    SubdivEvaluator(const Mesh &mesh);
    SubdivEvaluator(const SubdivEvaluator &that) { *this = that; }
    SubdivEvaluator &operator=(const SubdivEvaluator &that)
    {
        n_vertices = that.n_vertices;
        n_refiner_vertices = that.n_refiner_vertices;
        evaluation_verts_buffer = that.evaluation_verts_buffer;
        n_local_points = that.n_local_points;
        patch_table = new OpenSubdiv::Far::PatchTable(*that.patch_table);

        return *this;
    }

    std::shared_ptr<Mesh> generate_refined_mesh(const std::vector<Eigen::Vector3d> &verts_in, int level);
    void evaluate_subdiv_surface(const std::vector<Eigen::Vector3d> &coarse_verts, const std::vector<SurfacePoint> &uv, SurfaceFeatures &sf, const bool compute_dX = true) const;

    ~SubdivEvaluator()
    {
        delete patch_table;
        delete refiner;
    }
};
#pragma once

// required by LevenbergMarquardt file
#include <iostream>
using Scalar = double;

#include "mesh.h"
#include "subdiv_evaluator.h"
#include "visualizer.h"

#include <Eigen/Eigen>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/src/SparseExtra/BlockSparseQR.h>
#include <unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h>
#include <Open3D/Open3D.h>

#include <vector>
#include <filesystem>

struct FittingFunctor : Eigen::SparseFunctor<double>
{
    std::shared_ptr<std::vector<Eigen::Vector3d>> points;
    Mesh mesh;
    Mesh template_mesh;
    SubdivEvaluator evaluator;
    SurfaceFeatures sf;
    Visualizer* vis_ptr;
    int iteration;
    std::filesystem::path dir_path;

    // optimization variables
    struct InputType
    {
        Eigen::Vector3d scale;
        Eigen::Vector3d translation;
        std::vector<SurfacePoint> correspondences;

        InputType() {}

        InputType(const Eigen::Vector3d &scale, const Eigen::Vector3d &translation, const std::vector<SurfacePoint> &correspondences)
            : scale{scale}, translation{translation}, correspondences{correspondences}
        {
        }
    };

    FittingFunctor(const std::shared_ptr<std::vector<Eigen::Vector3d>> points_, const Mesh& mesh_, const Mesh& template_mesh_, Visualizer* vis_ptr, const std::filesystem::path dir_path_ = "")
        : SparseFunctor<double>{
              3 + 3 + static_cast<int>(points_->size()) * 2, // number of parameters = (scale + translation + correspondences)
              static_cast<int>(points_->size()) * 3},        // number of resiudals
          points{points_}, mesh{mesh_}, template_mesh{template_mesh_}, evaluator{mesh_}, sf{static_cast<int>(points_->size())}, vis_ptr{vis_ptr}, iteration{0}, dir_path{dir_path_}
    {
        vis_ptr->add_mesh(mesh.tri_mesh);
        vis_ptr->show();

        if (!dir_path.empty())
        {
            if (std::filesystem::exists(dir_path)) std::filesystem::remove_all(dir_path);
            std::filesystem::create_directory(dir_path);
            std::stringstream ss;
            ss << dir_path.string() << "/" << std::setw(2) << std::setfill('0') << iteration << ".png";
            vis_ptr->save_image(ss.str());
        }
    }

    // functor functions to be implemented
    int operator()(const InputType &x, ValueType &fvec);
    int df(const InputType &x, JacobianType &fjac);
    // int increment_u_crossing_edges(const std::vector<Eigen::Vector3d> &X, int &face, Eigen::Vector2d &u, const Eigen::Vector2d &du);
    void increment_in_place(InputType *x, const StepType &p);
    double estimateNorm(const InputType &x, const StepType &diag);

    /* describe QR solvers
    * J1 = [J11 0   0   ... 0
    *       0   J12 0   ... 0
    *                   ...
    *       0   0   0   ... J1N];
    * 
    * J = [J1 J2];
    */

    // QR for J1 subblocks is 3x2
    using DenseQRSolver3x2 = Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 3, 2>>;

    // QR for J1 is block diagonal
    using LeftSuperBlockSolver = Eigen::BlockDiagonalSparseQR<JacobianType, DenseQRSolver3x2>;

    // QR for J1'J2 is general dense
    using RightSuperBlockSolver = Eigen::ColPivHouseholderQR<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;

    // QR for J is concatenation of the above
    using SchurLikeQRSolver = Eigen::BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver>;

    using QRSolver = SchurLikeQRSolver;

    // tell algorithm how to set QR parameters
    void initQRSolver(QRSolver &qr)
    {
        qr.setBlockParams(points->size() * 2);
        qr.getLeftSolver().setSparseBlockParams(3, 2);
    }

    static int update_surface_point(const Mesh &mesh, SubdivEvaluator &evaluator, int &face, Eigen::Vector2d &u, const Eigen::Vector2d &du);
};
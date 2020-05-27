#include "visualizer.h"
#include "mesh.h"
#include "subdiv_evaluator.h"
#include "fitting_functor.h"

#include <Eigen/Eigen>
#include <Open3D/Open3D.h>

#include <iostream>
#include <random>
#include "timer.h"

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;

// creates a mesh using given parameters, and evaluates points on the surface
std::vector<Vector3d> evaluate_parametric_mesh(const Vector3d scale, const Vector3d &translation, const std::vector<SurfacePoint> &sps)
{
    Mesh mesh = Mesh::create_sphere();
    mesh.transform(scale, translation);
    SubdivEvaluator evaluator(mesh);
    SurfaceFeatures sf(sps.size());
    evaluator.evaluate_subdiv_surface(mesh.vertices(), sps, sf, false);
    return sf.S;
}

std::vector<SurfacePoint> offset_gt_correspondences(const Mesh &mesh, const std::vector<SurfacePoint> &sps_gt)
{
    std::vector<SurfacePoint> sps{sps_gt};
    SubdivEvaluator evaluator{mesh};

    std::random_device rd;
    std::mt19937 random_generator(rd());
    std::normal_distribution<double> dist_offset(0.0, 3.0);
    for (auto &sp : sps)
    {
        Vector2d du = {dist_offset(random_generator), dist_offset(random_generator)};
        FittingFunctor::update_surface_point(mesh, evaluator, sp.face, sp.u, du);
    }

    return sps;
}

void print_vector_in_single_line(const std::string &param_name, const Eigen::VectorXd &param)
{
    std::cout << param_name << ": [";
    for (int i{0}; i < param.size(); ++i)
    {
        std::cout << param(i);
        if (i != param.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

int main()
{
    // for drawing various geometries in 1 window
    Visualizer visualizer;

    // template mesh
    Mesh template_mesh{Mesh::create_sphere()};
    // visualizer.add_mesh(template_mesh.tri_mesh);

    // groundtruth parameters
    const Vector3d scale_gt{2, 1.5, 1.0};
    const Vector3d translation_gt{1.0, 2.0, 3.0};
    const int n_data{100};
    const std::vector<SurfacePoint> sps_gt = SurfacePoint::generate(n_data, template_mesh.n_triangles());

    // generate data with groundtruth parameters
    std::vector<Vector3d> points_observed = evaluate_parametric_mesh(scale_gt, translation_gt, sps_gt);
    visualizer.add_point_cloud(points_observed, {0, 0, 1});

    // init parameters
    Vector3d scale_init{1., 1., 1.};
    Vector3d translation_init{0.0, 0, 0};
    Mesh mesh{Mesh::create_sphere()};
    mesh.transform(scale_init, translation_init);
    visualizer.add_mesh(mesh.tri_mesh);

    // init correspondences
    std::vector<SurfacePoint> sps_init = offset_gt_correspondences(mesh, sps_gt);
    std::vector<Vector3d> points_init = evaluate_parametric_mesh(scale_init, translation_init, sps_init);
    // visualizer.add_point_cloud(points_init, {1, 0, 0});

    // optimize
    FittingFunctor::InputType params(scale_init, translation_init, sps_init);
    FittingFunctor fitting_functor(std::make_shared<std::vector<Vector3d>>(points_observed), mesh, template_mesh, &visualizer);
    Eigen::LevenbergMarquardt<FittingFunctor> lm(fitting_functor);
    lm.setVerbose(true);
    lm.setMaxfev(10);
    Timer timer;
    Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);
    std::cout << "\nOptimization took " << timer.elapsed() << "s\n";

    // log
    std::cout << "\nGroundtruth\n";
    print_vector_in_single_line("Scale", scale_gt);
    print_vector_in_single_line("Translation", translation_gt);

    std::cout << "\nInitial\n";
    print_vector_in_single_line("Scale", scale_init);
    print_vector_in_single_line("Translation", translation_init);

    std::cout << "\nOptimized\n";
    print_vector_in_single_line("Scale", params.scale);
    print_vector_in_single_line("Translation", params.translation);

    // hold visualizer
    std::cout << "\nPress `q` to close visualizer window\n";
    visualizer.run();

    return 0;
}
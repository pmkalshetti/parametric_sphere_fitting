#pragma once

#include <Eigen/Eigen>
#include <Open3D/Open3D.h>

#include <vector>
#include <iostream>
#include <memory>

struct Mesh
{
    std::shared_ptr<open3d::geometry::TriangleMesh> tri_mesh;
    Eigen::Matrix3Xi face_adj;

    Mesh() {}

    Mesh(const std::shared_ptr<open3d::geometry::TriangleMesh> triangle_mesh)
        : tri_mesh{triangle_mesh}
    {
        update_adjacencies();
    }

    void transform(const Eigen::Vector3d &scale = {1, 1, 1}, const Eigen::Vector3d &translation = {0, 0, 0})
    {
        Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
        transformation.diagonal().segment<3>(0) = scale;
        transformation.col(3).segment<3>(0) = translation;
        tri_mesh->Transform(transformation);
    }

    static Mesh create_sphere(const double radius = 1.0, const int resolution = 10)
    {
        std::shared_ptr<open3d::geometry::TriangleMesh> triangle_mesh_ptr = open3d::geometry::TriangleMesh::CreateSphere(radius, resolution);

        return Mesh{triangle_mesh_ptr};
    }

    std::vector<Eigen::Vector3d> &vertices() { return tri_mesh->vertices_; }
    const std::vector<Eigen::Vector3d> &vertices() const { return tri_mesh->vertices_; }
    std::vector<Eigen::Vector3i> &triangles() { return tri_mesh->triangles_; }
    const std::vector<Eigen::Vector3i> &triangles() const { return tri_mesh->triangles_; }
    const int n_vertices() const { return tri_mesh->vertices_.size(); }
    const int n_triangles() const { return tri_mesh->triangles_.size(); }

    void update_adjacencies();
};
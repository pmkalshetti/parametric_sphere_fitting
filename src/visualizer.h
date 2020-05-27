#pragma once

#include "mesh.h"

#include <Open3D/Open3D.h>
#include <Eigen/Eigen>

#include <vector>


class Visualizer
{
public:
    Visualizer()
    {
        vis.CreateVisualizerWindow();
        vis.GetRenderOption().ToggleMeshShowWireframe();
    }

    void add_point_cloud(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &color = {0, 0, 0})
    {
        open3d::geometry::PointCloud pcd(points);
        pcd.PaintUniformColor(color);
        vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(pcd));
    }

    void add_mesh(std::shared_ptr<open3d::geometry::TriangleMesh> &tri_mesh)
    {
        tri_mesh->ComputeVertexNormals();
        vis.AddGeometry(tri_mesh);
    }

    void update(std::shared_ptr<open3d::geometry::TriangleMesh> &tri_mesh)
    {
        tri_mesh->ComputeTriangleNormals();
        vis.UpdateGeometry(tri_mesh);
    }

    bool show()
    {
        return vis.PollEvents();
    }

    void save_image(const std::string& filename)
    {
        vis.CaptureScreenImage(filename);
    }

    void run()
    {
        vis.Run();
        vis.DestroyVisualizerWindow();
    }

private:
    open3d::visualization::Visualizer vis;
};
#include "mesh.h"


void Mesh::update_adjacencies()
{
    // find the adjacent faces to each face
    face_adj.resize(3, tri_mesh->triangles_.size());
    face_adj.fill(-1);

    for (int f{0}; f < tri_mesh->triangles_.size(); ++f)
    {
        for (int k{0}; k < 3; ++k)
        {
            // find the kth edge
            int k_next = (k + 1) % 3;
            int edge[2] = {tri_mesh->triangles_[f](k), tri_mesh->triangles_[f](k_next)};

            // find face that shares its reverse
            int found = 0;
            int other = -1;
            for (int fa{0}; fa < tri_mesh->triangles_.size(); ++fa)
            {
                if (f == fa) continue;

                for (int l = 0; l < 3; ++l)
                {
                    int l_next = (l + 1) % 3;
                    if ((tri_mesh->triangles_[fa](l) == edge[1]) && tri_mesh->triangles_[fa](l_next) == edge[0])
                    {
                        other = fa;
                        found++;
                        break;
                    }
                }
                if (found) break;
            }
            assert(found == 1);
            face_adj(k, f) = other;
        }
    }
}

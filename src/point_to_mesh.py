import psh
import numpy as np
import trimesh
from collections import namedtuple

voxel_data=namedtuple("VoxelData",["voxel_id","triangle_ids"])
def query_point(hashmap,point,minimums,spacing):
    return hashmap[np.floor((point-minimums)/spacing).astype(hashmap.int_type)]
def integerify_grid(gridpoints,spacing):
    minimums=np.min(gridpoints,0)
    integer_positions=((gridpoints+spacing/2-minimums)//spacing).astype(hashmap.int_type)
    #find and fix collisions
    int2float=dict()
    for idx,integer_pos in enumerate(integer_positions):
        if integer_pos not in int2float:
            int2float[integer_pos]=gridpoints[idx]
        else:
            direction=np.sign(gridpoints[idx]-int2float[integer_pos])

def main(spacing=0.001):
    #make a voxel grid covering the mesh
    mesh=trimesh.load("~/catkin_ws/src/EDKC/HingeAssets/AsMachinedCorrectedInertiaSimplified/hinge_base.STL")
    gridpoints=mesh.bounding_box.sample_grid(step=spacing)#minimum corners
    minimums=np.min(gridpoints,0)
    integer_positions=np.floor((gridpoints+spacing/2-minimums)/spacing).astype(np.uint64)
    domain_width=np.max(integer_positions,0)-np.min(integer_positions,0)+1
    data=[]

    #compute which triangles of the mesh intersect each voxel
    collision_manager=trimesh.collision.CollisionManager()
    collision_manager.add_object("mesh",mesh)
    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    intersecting_voxels=set()
    for i,pt in enumerate(gridpoints):
        #get all triangles that intersect the voxel
        tf=np.eye(4)
        tf[:3,3]=pt
        intersects,contacts=collision_manager.in_collision_single(voxel,transform=tf,return_data=True)
        if intersects:
            element=psh.data_tuple(integer_positions[i],voxel_data(i,tuple(cd.index("mesh") for cd in contacts)))
            data.append(element)
            intersecting_voxels.add(tuple(integer_positions[i]))

    #store voxels with non-empty intersections in hashmap
    hashmap=psh.PerfectSpatialHashMap(data,3,domain_width,0,verbose=True)

    #test that query points inside the voxel are mapped to the correct voxel
    n_test_pts=5
    rng=np.random.default_rng(0)
    imaginary_pts=[]
    missing_pts=[]
    wrong_pts=[]
    for i,pt in enumerate(gridpoints):
        lambdas=rng.uniform(size=(n_test_pts,3))
        sample_points=lambdas*spacing+pt
        for j in range(n_test_pts):
            integer_coordinate=tuple(np.floor((sample_points[j]-minimums)/spacing).astype(np.uint64))
            assert(integer_coordinate==tuple(integer_positions[i]))
            try:
                element=query_point(hashmap,sample_points[j],minimums,spacing)
            except KeyError:
                if integer_coordinate in intersecting_voxels:
                    missing_pts.append(sample_points[j])
                element=None
            if element is not None:
                if integer_coordinate not in intersecting_voxels:
                    #imaginary point appeared
                    imaginary_pts.append(sample_points[j])
                elif element.voxel_id!=i:
                    #mapped to wrong voxel!
                    wrong_pts.append(sample_points[j])
    print("missing ("+str(len(missing_pts))+")")
    print("Extra ("+str(len(imaginary_pts))+")")
    print("Wrong ("+str(len(wrong_pts))+")")
    return hashmap, data, mesh,spacing,minimums,missing_pts,imaginary_pts,wrong_pts

def main_fixed_domain_width(count=100):
    #make a voxel grid covering the mesh
    mesh=trimesh.load("~/catkin_ws/src/EDKC/HingeAssets/AsMachinedCorrectedInertiaSimplified/hinge_base.STL")
    gridpoints=mesh.bounding_box.sample_grid(count=count)#minimum corners
    minimums=np.min(gridpoints,0)
    maximums=np.max(gridpoints,0)
    spacing=(maximums-minimums)/(count-1)
    integer_positions=np.floor((gridpoints+spacing/2-minimums)/spacing).astype(np.uint64)
    domain_width=np.max(integer_positions,0)-np.min(integer_positions,0)+1
    data=[]

    #compute which triangles of the mesh intersect each voxel
    collision_manager=trimesh.collision.CollisionManager()
    collision_manager.add_object("mesh",mesh)
    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    intersecting_voxels=set()
    for i,pt in enumerate(gridpoints):
        #get all triangles that intersect the voxel
        tf=np.eye(4)
        tf[:3,3]=pt
        intersects,contacts=collision_manager.in_collision_single(voxel,transform=tf,return_data=True)
        if intersects:
            element=psh.data_tuple(integer_positions[i],voxel_data(i,tuple(cd.index("mesh") for cd in contacts)))
            data.append(element)
            intersecting_voxels.add(tuple(integer_positions[i]))

    #store voxels with non-empty intersections in hashmap
    hashmap=psh.PerfectSpatialHashMap(data,3,domain_width,0,verbose=True)

    #test that query points inside the voxel are mapped to the correct voxel
    n_test_pts=5
    rng=np.random.default_rng(0)
    imaginary_pts=[]
    missing_pts=[]
    wrong_pts=[]
    for i,pt in enumerate(gridpoints):
        lambdas=rng.uniform(size=(n_test_pts,3))
        sample_points=lambdas*spacing+pt
        for j in range(n_test_pts):
            integer_coordinate=tuple(np.floor((sample_points[j]-minimums)/spacing).astype(np.uint64))
            assert(integer_coordinate==tuple(integer_positions[i]))
            try:
                element=query_point(hashmap,sample_points[j],minimums,spacing)
            except KeyError:
                if integer_coordinate in intersecting_voxels:
                    missing_pts.append(sample_points[j])
                element=None
            if element is not None:
                if integer_coordinate not in intersecting_voxels:
                    #imaginary point appeared
                    imaginary_pts.append(sample_points[j])
                elif element.voxel_id!=i:
                    #mapped to wrong voxel!
                    wrong_pts.append(sample_points[j])
    print("missing ("+str(len(missing_pts))+")")
    print("Extra ("+str(len(imaginary_pts))+")")
    print("Wrong ("+str(len(wrong_pts))+")")
    return hashmap, data, mesh,spacing,minimums,missing_pts,imaginary_pts,wrong_pts

def manual_grid(spacing=0.01):
    #make a voxel grid covering the mesh
    mesh=trimesh.load("~/catkin_ws/src/EDKC/HingeAssets/AsMachinedCorrectedInertiaSimplified/hinge_base.STL")
    maximums=mesh.bounding_box.extents/2+mesh.bounding_box.transform[:3,3]
    minimums=-mesh.bounding_box.extents/2+mesh.bounding_box.transform[:3,3]
    gridvectors=tuple(np.arange(minimums[i],maximums[i],spacing) for i in range(3))
    domain_width=np.array([len(gv) for gv in gridvectors])
    npts=np.prod(domain_width)
    gridpoints=np.stack(np.meshgrid(*gridvectors),axis=-1).reshape(npts,3)
    integer_positions=np.stack(np.meshgrid(*tuple(np.arange(len(gv)) for gv in gridvectors)),axis=-1).reshape(npts,3)
    data=[]

    #compute which triangles of the mesh intersect each voxel
    collision_manager=trimesh.collision.CollisionManager()
    collision_manager.add_object("mesh",mesh)
    tf_putting_corner_at_0=np.eye(4)
    tf_putting_corner_at_0[:3,3]=spacing/2*np.ones((3,))
    voxel=trimesh.primitives.Box(extents=spacing*np.ones((3,)),transform=tf_putting_corner_at_0)
    intersecting_voxels=set()
    for i,pt in enumerate(gridpoints):
        #get all triangles that intersect the voxel
        tf=np.eye(4)
        tf[:3,3]=pt
        intersects,contacts=collision_manager.in_collision_single(voxel,transform=tf,return_data=True)
        if intersects:
            element=psh.data_tuple(integer_positions[i],voxel_data(i,tuple(cd.index("mesh") for cd in contacts)))
            data.append(element)
            intersecting_voxels.add(tuple(integer_positions[i]))

    #store voxels with non-empty intersections in hashmap
    hashmap=psh.create_psh(data,3,domain_width,0,verbose=True)

    #test that query points inside the voxel are mapped to the correct voxel
    n_test_pts=5
    rng=np.random.default_rng(0)
    imaginary_pts=[]
    missing_pts=[]
    wrong_pts=[]
    for i,pt in enumerate(gridpoints):
        lambdas=rng.uniform(size=(n_test_pts,3))
        sample_points=lambdas*spacing+pt
        for j in range(n_test_pts):
            integer_coordinate=tuple(np.floor((sample_points[j]-minimums)/spacing).astype(np.uint64))
            assert(integer_coordinate==tuple(integer_positions[i]))
            try:
                element=query_point(hashmap,sample_points[j],minimums,spacing)
            except KeyError:
                if integer_coordinate in intersecting_voxels:
                    missing_pts.append(sample_points[j])
                element=None
            if element is not None:
                if integer_coordinate not in intersecting_voxels:
                    #imaginary point appeared
                    imaginary_pts.append(sample_points[j])
                elif element.voxel_id!=i:
                    #mapped to wrong voxel!
                    wrong_pts.append(sample_points[j])
    print("missing ("+str(len(missing_pts))+")")
    print("Extra ("+str(len(imaginary_pts))+")")
    print("Wrong ("+str(len(wrong_pts))+")")
    return hashmap, data, mesh,spacing,minimums,missing_pts,imaginary_pts,wrong_pts
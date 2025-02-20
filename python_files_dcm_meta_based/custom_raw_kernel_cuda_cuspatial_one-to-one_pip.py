import cupy as cp
import cuspatial
import cudf
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import time

# -------------------------------
# 🔹 Fixed CUDA Kernel
# -------------------------------
one_to_one_pip_kernel = cp.RawKernel(r'''
extern "C" __global__
void one_to_one_pip(const double* px, const double* py,
                    const double* poly_x, const double* poly_y,
                    const long long int* poly_part_offsets, // ✅ FIX: Use long long int*
                    int* results, int num_points) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return; // Prevent out-of-bounds execution

    double x = px[i];
    double y = py[i];

    // ✅ Validate offsets before using
    long long ring_start = poly_part_offsets[i];  // ✅ Correct type
    long long ring_end = poly_part_offsets[i + 1];

    printf("[Thread %d] ✅ ring_start=%lld, ring_end=%lld for Point (%.2f, %.2f)\n", i, ring_start, ring_end, x, y);

    if (ring_end <= ring_start || ring_start < 0) {
        printf("[Thread %d] ❌ ERROR: Invalid ring indices: ring_start=%lld, ring_end=%lld\n", i, ring_start, ring_end);
        results[i] = 0;
        return;
    }

    // 🔹 Point-in-Polygon Test
    bool inside = false;
    for (long long j = ring_start, k = ring_end - 1; j < ring_end; k = j++) {
        double xj = poly_x[j], yj = poly_y[j];
        double xk = poly_x[k], yk = poly_y[k];

        printf("[Thread %d] Edge from (%.2f, %.2f) to (%.2f, %.2f)\n", i, xj, yj, xk, yk);

        bool intersect = ((yj > y) != (yk > y)) &&
                         (x < (xk - xj) * (y - yj) / (yk - yj) + xj);

        if (intersect) {
            inside = !inside;
            printf("[Thread %d] 🔥 Intersection detected! Inside flipped to %d\n", i, inside);
        }
    }

    results[i] = inside ? 1 : 0;
    printf("[Thread %d] ✅ Final result: %d\n", i, results[i]);
}

''', 'one_to_one_pip')



def one_to_one_point_in_polygon(points_gs, polygons_gs):
    num_points = len(points_gs)

    # ✅ Extract CuPy arrays
    points_x = points_gs.points.x.to_cupy()
    points_y = points_gs.points.y.to_cupy()

    # ✅ Extract polygons properly
    polygons_gpd = polygons_gs.to_geopandas()

    poly_x = cp.concatenate([cp.array(p.exterior.xy[0]) for p in polygons_gpd], axis=0)
    poly_y = cp.concatenate([cp.array(p.exterior.xy[1]) for p in polygons_gpd], axis=0)

    # ✅ Compute proper offsets
    vertex_counts = [len(p.exterior.xy[0]) for p in polygons_gpd]
    poly_part_offsets = cp.array([0] + vertex_counts, dtype=cp.int64).cumsum()  # ✅ Fix type

    # ✅ Debugging print statements
    print("\n🔹 Debugging Python Extraction")
    print("Points X:", points_x)
    print("Points Y:", points_y)
    print("Polygon X:", poly_x)
    print("Polygon Y:", poly_y)
    print("Polygon Part Offsets:", poly_part_offsets)

    # ✅ Allocate GPU memory for results
    results = cp.zeros(num_points, dtype=cp.int32)

    # ✅ Launch Kernel
    block_size = 256
    grid_size = (num_points + block_size - 1) // block_size
    one_to_one_pip_kernel((grid_size,), (block_size,), (
        points_x, points_y, poly_x, poly_y, poly_part_offsets, results, num_points
    ))

    # ✅ Retrieve results
    results_host = results.get()
    print("\n🔹 CUDA Results Host:", results_host)

    return results

"""
# -------------------------------
# 🔹 Example Usage
# -------------------------------
points_list = [Point(0.5, 0.5), Point(1.5, 1.5), Point(3.5, 3.5)]
polygons_list = [
    Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)]),  # Point (0.5, 0.5) is inside
    Polygon([(1,1), (2,1), (2,2), (1,2), (1,1)]),  # Point (1.5, 1.5) is inside
    Polygon([(2,2), (3,2), (3,3), (2,3), (2,2)])   # Point (3.5, 3.5) is outside
]

# ✅ Convert to `cuspatial.GeoSeries`
polygons_gs = cuspatial.GeoSeries(gpd.GeoSeries(polygons_list))
points_gs = cuspatial.GeoSeries(gpd.GeoSeries(points_list))

# ✅ Run Fixed Kernel
one_to_one_results = one_to_one_point_in_polygon(points_gs, polygons_gs)
print("\n🔹 Optimized One-to-One Results:")
print(one_to_one_results)
"""



# -------------------------------
# 🔹 Generate 10,000 Random Points & Polygons
# -------------------------------
num_points = 1000

# Generate random points inside (0,3)x(0,3)
random_points = [Point(x, y) for x, y in zip(np.random.uniform(0, 3, num_points), 
                                             np.random.uniform(0, 3, num_points))]

# Generate corresponding polygons around slightly different random locations
random_polygons = [
    Polygon([(x-0.1, y-0.1), (x+0.1, y-0.1), (x+0.1, y+0.1), (x-0.1, y+0.1), (x-0.1, y-0.1)]) 
    for x, y in zip(np.random.uniform(0, 3, num_points), np.random.uniform(0, 3, num_points))
]

# ✅ Convert to `cuspatial.GeoSeries`
polygons_gs = cuspatial.GeoSeries(gpd.GeoSeries(random_polygons))
points_gs = cuspatial.GeoSeries(gpd.GeoSeries(random_points))

# -------------------------------
# 🔹 Run Custom One-to-One Test
# -------------------------------
st = time.time()
one_to_one_results = one_to_one_point_in_polygon(points_gs, polygons_gs)
et= time.time()
print("\n🔹 Custom One-to-One Time:", et-st)

# -------------------------------
# 🔹 Run Default `cuspatial.point_in_polygon` in Chunks (Max 31 Polygons per Batch)
# -------------------------------
def chunked_point_in_polygon(points_gs, polygons_gs, chunk_size=31):
    num_polygons = len(polygons_gs)
    num_points = len(points_gs)
    
    # ✅ Allocate a results matrix
    results_matrix = cp.zeros((num_points, num_polygons), dtype=cp.bool_)

    for start in range(0, num_polygons, chunk_size):
        end = min(start + chunk_size, num_polygons)
        print(f"🔹 Processing polygons {start}-{end}")

        # ✅ Extract the chunk of polygons
        polygons_chunk = polygons_gs[start:end]

        # ✅ Run cuSpatial's `point_in_polygon` on this chunk
        chunk_results = cuspatial.point_in_polygon(points_gs, polygons_chunk).to_cupy()

        # ✅ Store the results in the full matrix
        results_matrix[:, start:end] = chunk_results

    return results_matrix

# 🔹 Run Chunked Default Test
st = time.time()
default_results = chunked_point_in_polygon(points_gs, polygons_gs)
et = time.time()
print("\n🔹 Default CuSpatial Time (Chunked):", et - st)

# Extract diagonal results from default `point_in_polygon`
diagonal_results = cp.diag(default_results)

# -------------------------------
# 🔹 Compare Results
# -------------------------------
print("\n🔹 Optimized One-to-One Results:")
print(one_to_one_results[0:20])  # Print first 20 results

print("\n🔹 Diagonal Results from Default CuSpatial:")
print(diagonal_results[0:20])  # Print first 20 results

# Check if results match
print("\n✅ Do Results Match?", cp.all(one_to_one_results == diagonal_results))

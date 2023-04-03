import taichi as ti
import numpy as np

@ti.data_oriented
class ParticleSystem:
    def __init__(self, resolution):
        self.resolution = resolution
        self.dim = len(resolution)
        assert self.dim > 1 & self.dim < 4

        # material
        self.material_boundary = 0
        self.materila_fluid = 1

        # particle settings
        self.particle_radius = 0.05
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius*4.0
        self.m_V = 0.8 * self.particle_diameter**self.dim
        self.particle_max_num = 2**15
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100
        self.particle_num = ti.field(ti.i32, shape=())

        # grid information
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(np.array(resolution)/self.grid_size).astype(int)
        self.grid_particles_num = ti.field(ti.i32)
        self.grid_particles = ti.field(ti.i32)

        # particle information
        self.x = ti.Vector.field(n=self.dim, dtype=ti.f32)
        self.v = ti.Vector.field(n=self.dim, dtype=ti.f32)
        self.density = ti.field(dtype=ti.f32)
        self.pressure = ti.field(dtype=ti.f32)
        self.material = ti.field(dtype=ti.i32)
        self.color = ti.field(dtype=ti.i32)
        self.particle_neighbors_num = ti.field(dtype=ti.i32)
        self.particle_neighbors = ti.field(dtype=ti.i32)

        # initialize
        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        self.particles_node.place(self.x, self.v, self.density, self.pressure, self.material, self.color)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbor)
        self.particle_node.place(self.particle_neighbors)

        index = ti.ij if self.dim == 2 else ti.ijk
        grid_node = ti.root.dense(index, self.grid_num)
        grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)
        cell_node.place(self.grid_particles)

    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_color[p - self.particle_num[None]])
        self.particle_num[None] += new_particles_num
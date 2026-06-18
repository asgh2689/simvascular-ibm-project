import numpy as np
import matplotlib.pyplot as plt

#Grid and domain sizes
nx, ny = 40, 40
lx, ly = 2.0, 2.0

#Coordinate matrices generated using 1D arrays
array_x = np.linspace(0, lx, nx)
array_y = np.linspace(0, ly, ny)
mesh_x, mesh_y = np.meshgrid(array_x, array_y, indexing='xy')

#Creating velocity fields
initialize_u = np.ones((ny,nx))
initialize_v = np.zeros((ny, nx))

#Pressure matrix
p = np.zeros((ny, nx))

#Boundary masks
check_wall = np.zeros((ny, nx), dtype=bool)
check_wall[0, :] = True
check_wall[-1, :] = True

#Constants
fp = np.array([0.0, -5.0])
dt = 0.0001
frame_steps = 10000
xp, yp = 1.0, 0.5
kinematic_viscosity = 0.1
pressure_loop = 750


#Copy of initial u and v to allow loop functionality
u = initialize_u.copy()
v = initialize_v.copy()

#For plotting initial valve location
xp_start, yp_start = xp, yp

#Add tolerance to decrease divergence
tolerance = 1e-5

for steps in range(frame_steps):

    #Make sure the u and v grids start at 40x40 every time. We are doing calculations inside the grid to prevent "spillage" to the outside of the grid
    u_star = u.copy()
    v_star = v.copy()

    #Temporary arrays for vertical and horizontal forces which are updated after frame loops
    temporary_x_array = np.zeros((ny, nx))
    temporary_y_array = np.zeros((ny, nx))

    #Calculates size of grid cell
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    #Determines index position of valve in grid (column and row). Which pixel box is the valve located in?
    idx_x = int(xp/dx)
    idx_y = int(yp/dy)

    #Relative distance percentage. The percentage across that specific box (how many steps have you taken in that box?)
    tx = (xp - (idx_x * dx)) / dx
    ty = (yp - (idx_y * dy)) / dy

    #Coupling operators that acts to determine how much force is distributed between the valve and the fluid (distribution percentage)
    bottom_left_weight = (1 - tx) * (1 - ty)
    bottom_right_weight =  tx * (1 - ty)
    top_left_weight =  (1 - tx) * ty
    top_right_weight =  tx * ty

    #Calculate force matrices
    temporary_x_array[idx_y, idx_x] += bottom_left_weight * fp[0]
    temporary_y_array[idx_y, idx_x] += bottom_left_weight * fp[1]
    temporary_x_array[idx_y, idx_x + 1] += bottom_right_weight * fp[0]
    temporary_y_array[idx_y, idx_x + 1] += bottom_right_weight * fp[1]
    temporary_x_array[idx_y + 1, idx_x + 1] += top_right_weight * fp[0]
    temporary_y_array[idx_y + 1, idx_x + 1] += top_right_weight * fp[1]
    temporary_x_array[idx_y + 1, idx_x] += top_left_weight * fp[0]
    temporary_y_array[idx_y + 1, idx_x] += top_left_weight * fp[1]

    #Calculate movement of fluid, ignoring incompressibility for now (part of Navier Stokes)
    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (kinematic_viscosity * ((u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx ** 2 + 
                                                         (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy ** 2) + temporary_x_array[1:-1, 1:-1])
    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (kinematic_viscosity * ((v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx ** 2 + 
                                                         (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy ** 2) + temporary_y_array[1:-1, 1:-1])
    
    #How much net fluid is entering or leaving (divergence)
    divergence = (((u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * dx)) + ((v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * dy))) / dt

    if steps % 100 == 0:
        print(f"Divergence before pressure loop: {np.max(np.abs(divergence)) / 10000:.4f}")

    #Pressure loop
    for pressure in range(pressure_loop):
        #Temporary pressure array
        p_temporary = p.copy()

        #Calculate pressure using poisson's equation
        p[1:-1,1:-1] = (((p_temporary[1:-1, 2:] + p_temporary[1:-1, :-2]) * dy**2 + (p_temporary[2:, 1:-1] + p_temporary[:-2, 1:-1]) * dx**2 -
                        divergence * dx**2 * dy**2) / (2 * (dx**2 + dy**2)))
        
        #Minimize divergence
        pressure_change = np.max(np.abs(p - p_temporary))
        if pressure_change < tolerance:
            break

    #Enforce boundary conditions to set pressure to 0 at walls
    p[0, :] = 0
    p[ny-1, :] = 0
    p[:, 0] = 0
    p[:, nx-1] = 0

    #Calculate velocity gradients using central difference
    u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - (dt * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx))
    v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - (dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy))

    #No-slip conditions at wall
    u[check_wall] = 0
    v[check_wall] = 0

    #Velocity of valve when fluid interacts with it
    u_valve = (u[idx_y, idx_x] * bottom_left_weight) + (u[idx_y, idx_x + 1] * bottom_right_weight) + (u[idx_y + 1, idx_x + 1] * top_right_weight) + (u[idx_y + 1, idx_x] * top_left_weight)
    v_valve = (v[idx_y, idx_x] * bottom_left_weight) + (v[idx_y, idx_x + 1] * bottom_right_weight) + (v[idx_y + 1, idx_x + 1] * top_right_weight) + (v[idx_y + 1, idx_x] * top_left_weight)

    #Update valve position using interpolated fluid velocities
    xp += u_valve * dt
    yp += v_valve * dt

    #Keep the valve inside the grid where interpolation is valid
    xp = np.clip(xp, dx, lx - 1.01 * dx)
    yp = np.clip(yp, dy, ly - 1.01 * dy)

    #Check if physical velocity field is incompressible
    post_divergence = (((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) + ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)))

    #Remove outer boundary 
    in_divergence = post_divergence[2:-2, 2:-2]

    if steps % 100 == 0:
        print(f"Frame {steps}")
        print(f"Divergence with walls: {np.max(np.abs(post_divergence)):.4f}")
        print(f"Divergence with fluid:   {np.max(np.abs(in_divergence)):.4f}")
        print(f"Max u: {np.max(np.abs(u)):.4f} | Max v: {np.max(np.abs(v)):.4f}")
        print(f"Valve position: ({xp:.3f}, {yp:.3f}) | valve velocity: ({u_valve:.3f}, {v_valve:.3f})")

plt.figure(figsize=(8, 8))
plt.quiver(mesh_x[::2, ::2], mesh_y[::2, ::2], u[::2, ::2], v[::2, ::2], color='blue', scale=20, pivot='mid')
plt.plot(xp, yp, 'ro', markersize=10, label='Final valve position')
plt.plot(xp_start, yp_start, 'go', markersize=10, label='Initial valve position')
plt.xlim(0, lx)
plt.ylim(0, ly)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'FSI Solver Frame: {frame_steps}')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
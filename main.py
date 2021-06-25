import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


# get dots a and a
# returns ab vector
def get_vec(a, b):
    vec = np.zeros(len(a))
    for i in range(len(a)):
        vec[i] = b[i] - a[i]
    return vec

# gets vectors a and b adn dot A
# returns plane [N, d], where N is normal vector and d = (N, X) where X is any dot of plane
def get_plane(a, b, A):
    # vector multiply
    N = np.cross(a, b)
    # scalar multiply
    d = np.inner(N, A)
    return [N, d]


# returns True if vectors x and y is collinear, else False
def check_collin(x, y):
    # if x or y is 0 vector
    if np.all(x == 0) or np.all(y == 0):
        return True
    i = 0
    beta = 0
    while i < len(x):
        if y[i] == 0 or x[i] == 0:
            i += 1
        else:
            beta = float(x[i]) / y[i]
            break
    # if x and y is not 0 vectors and beta is 0 then there will be at least one i: x[i] != beta * y[i]
    if beta == 0:
        return False
    # normal case
    for i in range(i, len(x)):
        if x[i] != beta * y[i]:
            return False
    return True


def task_dot_in_line():
    # dots A, B, C
    dot_a = np.array([7, 5])
    dot_b = np.array([4, 4])
    dot_c = np.array([1, 9])
    # if C is in line AB, then vectors AB and AC is collinear
    check = check_collin(get_vec(dot_a, dot_b), get_vec(dot_a, dot_c))
    print(check)
    # draws task in 2D
    draw_task_dot_in_line_2D(dot_a, dot_b, dot_c, check)

# draws smooth arrow
def draw_vec_2D(a, b, ax):

    vec_ab = get_vec(a, b)

    vec_ab_magnitude = math.sqrt(vec_ab[0] ** 2 + vec_ab[1] ** 2)

    if vec_ab_magnitude == 0:
        return 0

    head_length = 0.7

    vec_ab[0] = vec_ab[0] / vec_ab_magnitude
    vec_ab[1] = vec_ab[1] / vec_ab_magnitude

    vec_ab_magnitude = vec_ab_magnitude - head_length

    ax.arrow(a[0], a[1], vec_ab_magnitude * vec_ab[0], vec_ab_magnitude * vec_ab[1], head_width=0.5, head_length=0.7, fc='lightblue',
             ec='black')


# draws task_dot_in_line in 2D
def draw_task_dot_in_line_2D(a, b, c, check):

    ax = plt.axes()
    plt.grid()

    draw_vec_2D(a, b, ax)
    draw_vec_2D(a, c, ax)

    ax.annotate('A' + str(a), (a[0] - 0.05, a[1] + 0.2), fontsize=14)
    ax.annotate('B' + str(b), (b[0] - 0.05, b[1] + 0.2), fontsize=14)
    ax.annotate('C' + str(c), (c[0] - 0.05, c[1] + 0.2), fontsize=14)
    plt.scatter(a[0], a[1], color="blue")
    plt.scatter(b[0], b[1], color="cyan")
    plt.scatter(c[0], c[1], color="red")

    if check:
        ax.annotate('C is in line AB', (3.6, 1), fontsize=14)
    else:
        ax.annotate('C is NOT in line AB', (3, 1), fontsize=14)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()


def task_dot_in_plane():
    # dots A, B, C, D
    dot_a = np.array([1, 2, 3])
    dot_b = np.array([3, 2, 1])
    dot_c = np.array([5, 5, 5])
    dot_d = np.array([8, 5, 6])

    # gets plain ABC
    plane = get_plane(get_vec(dot_a, dot_b), get_vec(dot_a, dot_c), dot_a)
    # check if plane ABC contains D
    check = check_dot_in_plane(plane, dot_d)
    print(check)
    draw_task_dot_in_plane(dot_a, dot_b, dot_c, dot_d, plane)


# returns True if d is in plane, formed by vectors a, b and dot A
def check_dot_in_plane(plane, d):
    if np.inner(plane[0], d) == plane[1]:
        return True
    else:
        return False


# draws task_dot_in_plane
def draw_task_dot_in_plane(a, b, c, d, plane):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(-10, 11, 5)
    y = np.arange(-10, 11, 5)
    X, Y = np.meshgrid(x, y)

    if plane[0][2] != 0:
        def get_z(x, y):
            return (plane[1] - plane[0][0]*x - plane[0][1]*y)/plane[0][2]
    else:
        def get_z(x, y):
            return y
    z = get_z(X, Y)

    if plane[0][2] != 0:
        ax.plot_surface(X, Y, z)
    else:
        ax.plot_surface(X, y, z)

    ax.scatter(a[0], a[1], a[2], color="red")
    ax.scatter(b[0], b[1], b[2], color="red")
    ax.scatter(c[0], c[1], c[2], color="red")
    ax.scatter(d[0], d[1], d[2], color="black")
    plt.show()


def task_mutual_arrangement_lines():
    # dots A, B, C
    dot_a = np.array([1, 2, 6])
    dot_b = np.array([3, 1, 4])
    dot_c = np.array([7, 4, 3])
    dot_d = np.array([1, 2, 1])
    check = check_mut_arr_line_line(dot_a, dot_b, dot_c, dot_d)
    print(check)
    draw_task_mutual_arrangement_lines(dot_a, dot_b, dot_c, dot_d)


def check_mut_arr_line_line(a, b, c, d):
    x = get_vec(a, b)
    y = get_vec(c, d)
    z = get_vec(a, c)
    # returns -1 if x or y is 0 vector
    if np.all(x == 0) or np.all(y == 0):
        return -1
    if check_collin(x, y):
        if check_collin(x, z) and check_collin(y, z):
            # returns 0 if x and y is same line
            return 0
        else:
            # returns 1 if x and y is parallel
            return 1
    else:
        if check_dot_in_plane(get_plane(x, z, a), d):
            # returns 2 if x intersect y
            return 2
        else:
            # returns 3 if x cross y
            return 3


def draw_task_mutual_arrangement_lines(a, b, c, d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # line ab is black
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="black")
    # line ac is blue
    ax.plot([a[0], c[0]], [a[1], c[1]], [a[2], c[2]], color="blue")
    # line cd is red
    ax.plot([c[0], d[0]], [c[1], d[1]], [c[2], d[2]], color="red")
    plt.show()


def task_mutual_arrangement_line_plane():
    # dots A, B, C
    ABC = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    # dots M, N
    MN = np.array([[5, 5, 5], [1, 1, 1]])

    # vector AB
    x = get_vec(ABC[0], ABC[1])
    # vector AC
    y = get_vec(ABC[0], ABC[2])
    plane = get_plane(x, y, ABC[0])

    check = check_mut_arr_line_plane(plane, MN)
    draw_task_mut_arr_line_plane(plane, MN)


def check_mut_arr_line_plane(plane, MN):
    # vector MN
    z = get_vec(MN[0], MN[1])
    # check using scalar multiply
    if np.inner(plane[0], z) != 0:
        # returns 0 if MN intersects ABC
        return 0
    else:
        # checks is M is in ABC
        if check_dot_in_plane(plane, MN[0]):
            # returns 1 if NM is in ABC
            return 1
        else:
            # returns 2 if NM is parallel to ABC
            return 2


def draw_task_mut_arr_line_plane(plane, MN):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(-10, 11, 5)
    y = np.arange(-10, 11, 5)
    X, Y = np.meshgrid(x, y)

    if plane[0][2] != 0:
        def get_z(x, y):
            return (plane[1] - plane[0][0]*x - plane[0][1]*y)/plane[0][2]
    else:
        def get_z(x, y):
            return y
    z = get_z(X, Y)

    if plane[0][2] != 0:
        ax.plot_surface(X, Y, z)
    else:
        ax.plot_surface(X, y, z)

    ax.plot([MN[0][0], MN[1][0]], [MN[0][1], MN[1][1]], [MN[0][2], MN[1][2]], color="black")
    plt.show()


def task_mut_arr_planes():
    # dots A, B, C
    ABC = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    # dots L, M, N
    LMN = np.array([[0, 0, 0], [-1, -1, -1], [0, 0, -1]])
    plane_1 = get_plane(get_vec(ABC[0], ABC[1]), get_vec(ABC[0], ABC[2]), ABC[0])
    plane_2 = get_plane(get_vec(LMN[0], LMN[1]), get_vec(LMN[0], LMN[2]), LMN[0])
    check = check_mut_arr_planes(plane_1, plane_2, ABC[0])
    print(check)
    draw_mutual_arrangement_planes(plane_1, plane_2)


# A is point in plane_1
def check_mut_arr_planes(plane_1, plane_2, A):
    # if normalies is collinear
    if not check_collin(plane_1[0], plane_2[0]):
        # returns 0 if plane_1 intersect plane_2
        return 0
    else:
        # if A is in plane_2
        if check_dot_in_plane(plane_2, A):
            # returns 1 if plane_1 and plane_2 is same
            return 1
        else:
            # returns 2 if plane_1 and plane_2 is parallel
            return 2


def draw_mutual_arrangement_planes(plane_1, plane_2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(-10, 11, 5)
    y = np.arange(-10, 11, 5)
    X, Y = np.meshgrid(x, y)

    if plane_1[0][2] != 0:
        def get_z(x, y):
            return (plane_1[1] - plane_1[0][0]*x - plane_1[0][1]*y)/plane_1[0][2]
    else:
        def get_z(x, y):
            return y
    z_1 = get_z(X, Y)

    if plane_2[0][2] != 0:
        def get_z(x, y):
            return (plane_2[1] - plane_2[0][0]*x - plane_2[0][1]*y)/plane_2[0][2]
    else:
        def get_z(x, y):
            return y
    z_2 = get_z(X, Y)

    if plane_1[0][2] != 0:
        ax.plot_surface(X, Y, z_1)
    else:
        ax.plot_surface(X, y, z_1)

    if plane_2[0][2] != 0:
        ax.plot_surface(X, Y, z_2)
    else:
        ax.plot_surface(X, y, z_2)

    plt.show()


def task_get_seg_divider():
    # dots A, B, C
    dots = np.array([[0, 0, 0], [3, 6, 12], [1, 2, 4]])
    a = get_vec(dots[0], dots[2])
    b = get_vec(dots[2], dots[1])
    d = get_seg_divider(a, b)
    print(d)
    draw_task_seg_divider(dots[0], a, d)


# a is AC, b is CB
def get_seg_divider(a, b):
    # if a is 0 vector
    if np.all(a == 0):
        # returns -1 if C is same as A
        return -1
    # if b is 0 vector
    if np.all(b == 0):
        # returns 0 if C is same as B
        return 0
    i = 0
    d = 0
    # AC and CB is collinear, so there will be at least one i: a[i] != 0 and b[i] != 0
    # if there is more than one such i, we need only the first d as they will all be same
    while i < len(a):
        if a[i] == 0 or b[i] == 0:
            i += 1
        else:
            d = float(a[i]) / b[i]
            break
    return d


def draw_task_seg_divider(A, a, d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    C = A + a
    B = C + float(1) / d * a
    # line AC is black
    ax.plot([A[0], C[0]], [A[1], C[1]], [A[2], C[2]], color="black")
    # line CB is red
    ax.plot([C[0], B[0]], [C[1], B[1]], [C[2], B[2]], color="red")
    plt.show()




def task_seg_div():
    # dots A, B, O
    dots = np.array([[0, 0, 0], [2, 4, 6], [3, 3, 3]])
    # division relation: AC = d * CB
    d = -5
    OC = seg_div(get_vec(dots[2], dots[0]), get_vec(dots[2], dots[1]), d)
    print(OC)
    draw_task_seg_div(dots, OC)


#  returns vector OC from vectors OA, OB and division relation
# a and b are OA and OB, l is division relation
def seg_div(a, b, dr):
    return float(1)/(1+dr) * a + float(dr)/(1+dr) * b


# a is OC
def draw_task_seg_div(dots, a):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # line ab is black
    ax.plot([dots[0][0], dots[1][0]], [dots[0][1], dots[1][1]], [dots[0][2], dots[1][2]], color="black")
    C = dots[2] + a
    # line OC is red
    ax.plot([dots[2][0], C[0]], [dots[2][1], C[1]], [dots[0][2], C[2]], color="red")
    plt.show()


task_dot_in_line()
# task_dot_in_plane()
# task_mutual_arrangement_lines()
# task_mutual_arrangement_line_plane()
# task_mut_arr_planes()
# task_get_seg_divider()
# task_seg_div()
# task_get_seg_divider()

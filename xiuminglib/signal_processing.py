"""
Functions for Signal Processing Techniques

Xiuming Zhang, MIT CSAIL
August 2017
"""

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.special import sph_harm


def get(arr, top=True, n=1, n_std=None):
    """
    Get top (or bottom) n value(s) from an m-D array_like

    Args:
        arr: Array, which will be flattened if high-D
            m-D array_like
        top: Whether to find the top or bottom n
            Boolean
            Optional; defaults to True
        n: Number of values to return
            Positive integer
            Optional; defaults to 1
        n_std: Definition of outliers to exclude, assuming Gaussian
            Positive float
            Optional; defaults to None (assuming no outlier)

    Returns:
        ind: Indice(s) that give the extrema
            m-tuple of numpy arrays of n integers
        val: Extremum values, i.e., `arr[ind]`
            Numpy array of length n
    """
    arr = np.array(arr, dtype=float)

    if top:
        arr_to_sort = -arr.flatten()
    else:
        arr_to_sort = arr.flatten()

    if n_std is not None:
        meanv = np.mean(arr_to_sort)
        stdv = np.std(arr_to_sort)
        arr_to_sort[np.logical_or(
            arr_to_sort < meanv - n_std * stdv,
            arr_to_sort > meanv + n_std * stdv,
        )] = np.nan # considered greater than numbers

    ind = [x for x in np.argsort(arr_to_sort)
           if not np.isnan(arr_to_sort[x])][:n] # 1D indices
    ind = np.unravel_index(ind, arr.shape) # Back to high-D
    val = arr[ind]

    return ind, val


def smooth_1d(arr, win_size, kernel_type='half'):
    """
    Smooth 1D signal

    Args:
        arr: 1D signal to smooth
            1D array_like of floats
        win_size: Size of the smoothing window
            Odd natural number
        kernel_type: Kernel type
            'half' (e.g., normalized [2^-2, 2^-1, 2^0, 2^-1, 2^-2]) or
            'equal' (e.g., normalized [1, 1, 1, 1, 1]
            Optional (defaults to 'half')

    Returns:
        arr_smooth: Smoothed 1D signal
            1D numpy array of floats
    """
    assert np.mod(win_size, 2) == 1, "Even window size provided"
    arr = np.array(arr).ravel()

    # Generate kernel
    if kernel_type == 'half':
        kernel = np.array([2 ** x if x < 0 else 2 ** -x
                           for x in range(-int(win_size / 2), int(win_size / 2) + 1)])
    elif kernel_type == 'equal':
        kernel = np.ones(win_size)
    else:
        raise ValueError("Unidentified kernel type")
    kernel /= sum(kernel)
    n = (win_size - 1) // 2

    arr_pad = np.hstack((arr[0] * np.ones(n), arr, arr[-1] * np.ones(n)))
    arr_smooth = np.convolve(arr_pad, kernel, 'valid')

    # Restore original values of the head and tail
    arr_smooth[0] = arr[0]
    arr_smooth[-1] = arr[-1]

    return arr_smooth


def pca(data_mat, n_pcs=None, eig_method='scipy.sparse.linalg.eigsh'):
    """
    Perform principal component (PC) analysis on data via eigendecomposition of covariance matrix
        See unit_test() for example usages (incl. reconstructing data with top k PC's)

    Args:
        data_mat: Data matrix of n data points in the m-D space
            Array_like, dense or sparse, of shape (m, n); each column is a point
        n_pcs: Number of top PC's requested
            Positive integer < m
            Optional; defaults to m - 1
        eig_method: Method for eigendecomposition of the symmetric covariance matrix
            'numpy.linalg.eigh' or 'scipy.sparse.linalg.eigsh'
            Optional; defaults to 'scipy.sparse.linalg.eigsh'

    Returns:
        pcvars: PC variances (eigenvalues of covariance matrix) in descending order
            Numpy array of length n_pcs
        pcs: Corresponding PC's (normalized eigenvectors)
            Numpy array of shape (m, n_pcs); each column is a PC
        projs: Data points centered and then projected to the n_pcs-D PC space
            Numpy array of shape (n_pcs, n); each column is a point
        data_mean: Mean that can be used to recover raw data
            Numpy array of length m
    """
    if issparse(data_mat):
        data_mat = data_mat.toarray()
    else:
        data_mat = np.array(data_mat)
    # data_mat is NOT centered

    if n_pcs is None:
        n_pcs = data_mat.shape[0] - 1

    # ------ Compute covariance matrix of data

    covmat = np.cov(data_mat) # auto handles uncentered data
    # covmat is real and symmetric in theory, but may not be so due to numerical issues,
    # so eigendecomposition method should be told explicitly to exploit symmetry constraints

    # ------ Compute eigenvalues and eigenvectors

    if eig_method == 'scipy.sparse.linalg.eigsh':
        # Largest (in magnitude) n_pcs eigenvalues
        eig_vals, eig_vecs = eigsh(covmat, k=n_pcs, which='LM')
        # eig_vals in ascending order
        # eig_vecs columns are normalized eigenvectors

        pcvars = eig_vals[::-1] # descending
        pcs = eig_vecs[:, ::-1]

    elif eig_method == 'numpy.linalg.eigh':
        # eigh() prevents complex eigenvalues, compared with eig()
        eig_vals, eig_vecs = np.linalg.eigh(covmat)
        # eig_vals in ascending order
        # eig_vecs columns are normalized eigenvectors

        # FIXME: sometimes the eigenvalues are not sorted? Subnormals appear. All zero eigenvectors
        sort_ind = eig_vals.argsort() # ascending
        eig_vals = eig_vals[sort_ind]
        eig_vecs = eig_vecs[:, sort_ind]

        pcvars = eig_vals[:-(n_pcs + 1):-1] # descending
        pcs = eig_vecs[:, :-(n_pcs + 1):-1]

    else:
        raise NotImplementedError(eig_method)

    # ------ Center and then project data points to PC space

    data_mean = np.mean(data_mat, axis=1)
    data_mat_centered = data_mat - np.tile(data_mean.reshape(-1, 1), (1, data_mat.shape[1]))
    projs = np.dot(pcs.T, data_mat_centered)

    return pcvars, pcs, projs, data_mean


def matrix_for_discrete_fourier_transform(n):
    """
    Generate transform matrix for discrete Fourier transform (DFT) W
        To transform an image I, apply it twice: WIW
        See unit_test() for example usages

    Args:
        n: Signal length; this will be either image height or width if you are doing 2D DFT
            to an image, i.e., wmat_h.dot(im).dot(wmat_w).
            Natural number

    Returns:
        wmat: Transform matrix whose row i, when dotting with signal (column) vector, gives the
            coefficient for i-th Fourier component, where i < n
            Numpy complex array of shape (n, n)
    """
    col_ind, row_ind = np.meshgrid(range(n), range(n))

    omega = np.exp(-2 * np.pi * 1j / n)
    wmat = np.power(omega, col_ind * row_ind) / np.sqrt(n) # normalize so that unitary

    return wmat


def matrix_for_real_spherical_harmonics(l, n_lat, coord_convention='colatitude-azimuth', _check_orthonormality=False):
    """
    Generate transform matrix for discrete real spherical harmonic (SH) expansion
        See unit_test() for example usages

    Args:
        l: Up to which band (starting form 0); the number of harmonics is (l + 1) ** 2;
            in other words, all harmonics within each band (-l <= m <= l) are used
            Natural number
        n_lat: Number of discretization levels of colatitude (colatitude-azimuth convention; [0, pi]) or
            latitude (latitude-longitude convention; [-pi/2, pi/2]); with the same step size, n_azimuth
            will be twice as big, since azimuth (colatitude-azimuth convention; [0, 2pi]) or latitude
            (latitude-longitude convention; [-pi, pi]) spans 2pi
            Natural number
        coord_convention: Coordinate system convention to use
            'colatitude-azimuth' or 'latitude-longitude'
            Optional; defaults to 'colatitude-azimuth'

            Colatitude-azimuth convention / latitude-longitude convention

                3D
                                                               ^ z (colat = 0 / lat = pi/2)
                                                               |
                                                               |
                          (azi = 3pi/2 / lng = -pi/2) ---------+---------> y (azi = pi/2 / lng = pi/2)
                                                             ,'|
                                                           ,'  |
              (colat = pi/2, azi = 0 / lat = 0, lng = 0) x     | (colat = pi / lat = -pi/2)

                2D
                     (0, 0)                                  (pi/2, 0)
                        +----------->  (0, 2pi)                  ^ lat
                        |            azi                         |
                        |                                        |
                        |                        (0, -pi) -------+-------> (0, pi)
                        v colat                                  |        lng
                     (pi, 0)                                     |
                                                            (-pi/2, 0)

        _check_orthonormality: Whether to check orthonormality or not
            Boolean
            Internal use only and optional; defaults to False

    Returns:
        ymat: Transform matrix whose row i, when dotted with flattened image (column) vector,
            gives the coefficient for i-th harmonic, where i = (l + 1) * l + m; the spherical
            function to transform (in the form of 2D image indexed by two angles) should be
            flattened, with .ravel(), in row-major order: the row index varies the slowest,
            and the column index the quickest
            Numpy array of shape ((l + 1) ** 2, 2 * n_lat ** 2)
        areas_on_unit_sphere: Area of the unit sphere covered by each sample point; this is
            proportional to sine of colatitude and has nothing to do with azimuth/longitude;
            Used as weights for discrete summation to approximate continuous integration;
            Flattened also in row-major order
            Numpy array of length n_lat * (2 * n_lat)
    """
    # Generate the l and m values for each matrix location
    l_mat = np.zeros(((l + 1) ** 2, n_lat * 2 * n_lat))
    m_mat = np.zeros(l_mat.shape)
    i = 0
    for curr_l in range(l + 1):
        for curr_m in range(-curr_l, curr_l + 1):
            l_mat[i, :] = curr_l * np.ones(l_mat.shape[1])
            m_mat[i, :] = curr_m * np.ones(l_mat.shape[1])
            i += 1

    # Generate the two angles for each matrix location
    step_size = np.pi / n_lat
    if coord_convention == 'colatitude-azimuth':
        azis, colats = np.meshgrid(
            np.linspace(0 + step_size, 2 * np.pi - step_size, num=2 * n_lat, endpoint=True),
            np.linspace(0 + step_size, np.pi - step_size, num=n_lat, endpoint=True))
    elif coord_convention == 'latitude-longitude':
        lngs, lats = np.meshgrid(
            np.linspace(-np.pi + step_size, np.pi - step_size, num=2 * n_lat, endpoint=True),
            np.linspace(np.pi / 2 - step_size, -np.pi / 2 + step_size, num=n_lat, endpoint=True))
        colats = np.pi / 2 - lats
        azis = lngs
        azis[azis < 0] += 2 * np.pi
    else:
        raise NotImplementedError(coord_convention)

    # Evaluate (complex) SH at these locations
    colat_mat = np.tile(colats.ravel(), (l_mat.shape[0], 1))
    azi_mat = np.tile(azis.ravel(), (l_mat.shape[0], 1))
    ymat_complex = sph_harm(m_mat, l_mat, azi_mat, colat_mat)

    sin_colat = np.sin(colats.ravel())
    # Area on the unit sphere covered by each sample point, proportional to sin(colat)
    # Used as weights for discrete summation, approximating continuous integration
    areas_on_unit_sphere = 4 * np.pi * sin_colat / np.sum(sin_colat)

    # Verify orthonormality of SH's
    if _check_orthonormality:
        print("Verifying Orthonormality of Complex SH Bases")
        print("(l1, m1) and (l2, m2):\treal\timag")
        for l1 in range(l + 1):
            for m1 in range(-l1, l1 + 1, 1):
                i1 = l1 * (l1 + 1) + m1
                y1 = ymat_complex[i1, :]
                for l2 in range(l + 1):
                    for m2 in range(-l2, l2 + 1, 1):
                        i2 = l2 * (l2 + 1) + m2
                        y2 = ymat_complex[i2, :]
                        integral = np.conj(y1).dot(np.multiply(areas_on_unit_sphere, y2))
                        integral_real = np.real(integral)
                        integral_imag = np.imag(integral)
                        if np.isclose(integral_real, 0):
                            integral_real = 0
                        if np.isclose(integral_imag, 0):
                            integral_imag = 0
                        print("(%d, %d) and (%d, %d):\t%f\t%f" %
                              (l1, m1, l2, m2, integral_real, integral_imag))

    # Derive real SH's
    ymat_complex_real = np.real(ymat_complex)
    ymat_complex_imag = np.imag(ymat_complex)
    ymat = np.zeros(ymat_complex_real.shape)
    ind = m_mat > 0
    ymat[ind] = (-1) ** m_mat[ind] * np.sqrt(2) * ymat_complex_real[ind]
    ind = m_mat == 0
    ymat[ind] = ymat_complex_real[ind]
    ind = m_mat < 0
    ymat[ind] = (-1) ** m_mat[ind] * np.sqrt(2) * ymat_complex_imag[ind]

    if _check_orthonormality:
        print("Verifying Orthonormality of Real SH Bases")
        print("(l1, m1) and (l2, m2):\tvalue")
        for l1 in range(l + 1):
            for m1 in range(-l1, l1 + 1, 1):
                i1 = l1 * (l1 + 1) + m1
                y1 = ymat[i1, :]
                for l2 in range(l + 1):
                    for m2 in range(-l2, l2 + 1, 1):
                        i2 = l2 * (l2 + 1) + m2
                        y2 = ymat[i2, :]
                        integral = y1.dot(np.multiply(areas_on_unit_sphere, y2))
                        if np.isclose(integral, 0):
                            integral = 0
                        print("(%d, %d) and (%d, %d):\t%f" %
                              (l1, m1, l2, m2, integral))

    return ymat, areas_on_unit_sphere


def unit_test(func_name):
    # Unit tests and example usages

    if func_name == 'pca':
        pts = np.random.rand(5, 8) # 8 points in 5D

        # Find all principal components
        n_pcs = pts.shape[0] - 1
        _, pcs, projs, data_mean = pca(pts, n_pcs=n_pcs)

        # Reconstruct data with only the top two PC's
        k = 2
        pts_recon = pcs[:, :k].dot(projs[:k, :]) + \
            np.tile(data_mean, (projs.shape[1], 1)).T
        pdb.set_trace()

    elif func_name == 'matrix_for_discrete_fourier_transform':
        im = np.random.randint(0, 255, (8, 10))
        h, w = im.shape

        # Transform by my matrix
        dft_mat_col = matrix_for_discrete_fourier_transform(h)
        dft_mat_row = matrix_for_discrete_fourier_transform(w)
        coeffs = dft_mat_col.dot(im).dot(dft_mat_row)

        # Transform by numpy
        coeffs_np = np.fft.fft2(im) / (np.sqrt(h) * np.sqrt(w))

        print("%s: max. magnitude difference: %e" % (func_name, np.abs(coeffs - coeffs_np).max()))
        pdb.set_trace()

    elif func_name == 'matrix_for_real_spherical_harmonics':
        from visualization import matrix_as_heatmap

        ls = [10, 20]
        n_steps_theta = 500
        h, w = 30, 70

        # Black background with a white rectangle
        sph_func = np.zeros((n_steps_theta, 2 * n_steps_theta))
        left_top_corner = (np.random.randint(0, n_steps_theta), np.random.randint(0, 2 * n_steps_theta))
        sph_func[left_top_corner[0]:min(left_top_corner[0] + h, n_steps_theta),
                 left_top_corner[1]:min(left_top_corner[1] + w, 2 * n_steps_theta)] = 255
        matrix_as_heatmap(sph_func, outpath='../../test-output/orig.png')

        for l in ls:
            # Construct matrix for discrete real SH transform
            ymat, weights = matrix_for_real_spherical_harmonics(l, n_steps_theta, _check_orthonormality=False)

            # Analysis
            sph_func_1d = sph_func.ravel()
            coeffs = ymat.dot(np.multiply(weights, sph_func_1d))

            # Synthesis
            sph_func_1d_recon = ymat.T.dot(coeffs)
            sph_func_recon = sph_func_1d_recon.reshape(sph_func.shape)
            matrix_as_heatmap(sph_func_recon, outpath='../../test-output/recon_l%03d.png' % l)
        pdb.set_trace()

    else:
        raise NotImplementedError("Unit tests for %s" % func_name)


if __name__ == '__main__':
    import pdb

    func_to_test = 'matrix_for_real_spherical_harmonics'
    unit_test(func_to_test)

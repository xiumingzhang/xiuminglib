import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.special import sph_harm


def get(arr, top=True, n=1, n_std=None):
    """Gets top (or bottom) N value(s) from an M-D array.

    Args:
        arr (array_like): Array, which will be flattened if high-D.
        top (bool, optional): Whether to find the top or bottom N.
        n (int, optional): Number of values to return.
        n_std (float, optional): Definition of outliers to exclude, assuming Gaussian.
            ``None`` means assuming no outlier.

    Returns:
        tuple:
            - **ind** (*tuple*) -- Indices that give the extrema, M-tuple of arrays of N integers.
            - **val** (*numpy.ndarray*) -- Extremum values, i.e., ``arr[ind]``.
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
    """Smooths 1D signal.

    Args:
        arr (array_like): 1D signal to smooth.
        win_size (int): Size of the smoothing window. Use odd number.
        kernel_type (str, optional): Kernel type: ``'half'`` (e.g., normalized :math:`[2^{-2}, 2^{-1},
            2^0, 2^{-1}, 2^{-2}]`) or ``'equal'`` (e.g., normalized :math:`[1, 1, 1, 1, 1]`).

    Raises:
        ValueError: If kernel type is wrong.

    Returns:
        numpy.ndarray: Smoothed 1D signal.
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
    """Performs principal component (PC) analysis on data.

    Via eigendecomposition of covariance matrix. See :func:`main` for example usages,
    including reconstructing data with top K PCs.

    Args:
        data_mat (array_like): Data matrix of N data points in the M-D space, of shape
            M-by-N, where each column is a point.
        n_pcs (int, optional): Number of top PCs requested. ``None`` means :math:`M-1`.
        eig_method (str, optional): Method for eigendecomposition of the symmetric covariance matrix:
            ``'numpy.linalg.eigh'`` or ``'scipy.sparse.linalg.eigsh'``.

    Raises:
        NotImplementedError: If ``eig_method`` is not implemented.

    Returns:
        tuple:
            - **pcvars** (*numpy.ndarray*) -- PC variances (eigenvalues of covariance matrix)
              in descending order.
            - **pcs** (*numpy.ndarray*) -- Corresponding PCs (normalized eigenvectors), of shape
              M-by-``n_pcs``. Each column is a PC.
            - **projs** (*numpy.ndarray*) -- Data points centered and then projected to the
              ``n_pcs``-D PC space. Of shape ``n_pcs``-by-N. Each column is a point.
            - **data_mean** (*numpy.ndarray*) -- Mean that can be used to recover raw data. Of length M.
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
    """Generates transform matrix :math:`W` for discrete Fourier transform (DFT).

    To transform an image :math:`I`, apply it twice: :math:`WIW`.
    See :func:`main` for example usages.

    Args:
        n (int): Signal length. This will be either image height or width if you are doing 2D DFT
            to an image, i.e., ``wmat_h.dot(im).dot(wmat_w)``.

    Returns:
        numpy.ndarray: Transform matrix whose row :math:`i`, when dotting with signal (column) vector,
        gives the coefficient for the :math:`i`-th Fourier component, where :math:`i < N`.
        Of shape N-by-N.
    """
    col_ind, row_ind = np.meshgrid(range(n), range(n))

    omega = np.exp(-2 * np.pi * 1j / n)
    wmat = np.power(omega, col_ind * row_ind) / np.sqrt(n) # normalize so that unitary

    return wmat


def matrix_for_real_spherical_harmonics(l, n_lat, coord_convention='colatitude-azimuth', _check_orthonormality=False):
    r"""Generates transform matrix for discrete real spherical harmonic (SH) expansion.

    See :func:`main` for example usages.

    Args:
        l (int): Up to which band (starting form 0). The number of harmonics is :math:`(l+1)^2`.
            In other words, all harmonics within each band (:math:`-l\leq m\leq l`) are used.
        n_lat (int): Number of discretization levels of colatitude (for colatitude-azimuth convention; :math:`[0, \pi]`)
            or latitude (for latitude-longitude convention; :math:`[-\frac{\pi}{2}, \frac{\pi}{2}]`).
            With the same step size, ``n_azimuth`` will be twice as big, since azimuth (in colatitude-azimuth convention;
            :math:`[0, 2\pi]`) or latitude (in latitude-longitude convention; :math:`[-\pi, \pi]`) spans :math:`2\pi`.
        coord_convention (str, optional): Coordinate system convention to use: ``'colatitude-azimuth'``
            or ``'latitude-longitude'``. Colatitude-azimuth vs. latitude-longitude convention:

            .. code-block:: none

                3D
                                                   ^ z (colat = 0; lat = pi/2)
                                                   |
                          (azi = 3pi/2;            |
                           lng = -pi/2)   ---------+---------> y (azi = pi/2; lng = pi/2)
                                                 ,'|
                    (colat = pi/2, azi = 0;    ,'  |
                     lat = 0, lng = 0)        x    | (colat = pi; lat = -pi/2)

                2D
                     (0, 0)                                  (pi/2, 0)
                        +----------->  (0, 2pi)                  ^ lat
                        |            azi                         |
                        |                                        |
                        |                        (0, -pi) -------+-------> (0, pi)
                        v colat                                  |        lng
                     (pi, 0)                                     |
                                                            (-pi/2, 0)

        _check_orthonormality (bool, optional): Whether to check orthonormality or not.
            Intended for internal use.

    Raises:
        NotImplementedError: If the coordinate convention specified is not implemented.

    Returns:
        tuple:
            - **ymat** (*numpy.ndarray*) -- Transform matrix whose row :math:`i`, when dotted with flattened image
              (column) vector, gives the coefficient for :math:`i`-th harmonic, where :math:`i=l(l+1)+m`.
              The spherical function to transform (in the form of 2D image indexed by two angles) should be
              flattened, with ``.ravel()``, in row-major order: the row index varies the slowest,
              and the column index the quickest. Of shape ``((l + 1) ** 2, 2 * n_lat ** 2)``.
            - **areas_on_unit_sphere** (*numpy.ndarray*) -- Area of the unit sphere covered by each sample point.
              This is proportional to sine of colatitude and has nothing to do with azimuth/longitude.
              Used as weights for discrete summation to approximate continuous integration.
              Flattened also in row-major order. Of length ``n_lat * (2 * n_lat)``.
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


def main(func_name):
    """Unit tests that can also serve as example usage."""
    import pdb

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
    main('matrix_for_real_spherical_harmonics')

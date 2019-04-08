import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.special import sph_harm


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


def gen_dft_bases_1d(n, upto=None):
    """Generates 1D discrete Fourier transform (DFT) bases.

    Bases are rows of :math:`Y`, a symmetric matrix. The Fourier coefficients are simply
    given by :math:`Yx`. See :func:`main` for example usages.

    Args:
        n (int): Signal length.
        upto (int, optional): Up to how many bases. ``None`` means all.

    Returns:
        numpy.ndarray: Matrix whose row :math:`i`, when dotted with signal (column) vector,
        gives the coefficient for the :math:`i`-th Fourier component.
        Of shape ``(min(n, upto), n)``.
    """
    col_ind, row_ind = np.meshgrid(range(n), range(n))
    omega = np.exp(-2 * np.pi * 1j / n)
    wmat = np.power(omega, col_ind * row_ind) / np.sqrt(n) # normalize
    # so that unitary (i.e., energy-preserving)
    if upto is not None:
        wmat = wmat[:upto, :]
    return wmat


def gen_dft_bases_2d(h, w, upto_h=None, upto_w=None):
    r"""Generates 2D discrete Fourier transform (DFT) bases.

    Bases are rows of :math:`Y`. Input image :math:`X` should be flattened with
    :meth:`numpy.ndarray.ravel` into a vector :math:`x`. Then, the coefficients are
    just :math:`Yx`.

    See :func:`main` for example usages and how this is related to :func:`gen_dft_bases_1d`.

    Args:
        h (int): Image height.
        w
        upto_h (int, optional): Up to how many bases in the height dimension. ``None`` means all.
        upto_w

    Returns:
        numpy.ndarray: Matrix whose row :math:`i`, when dotted with the flattened input,
        gives the coefficient for the :math:`(i_h, i_w)`-th Fourier component, where
        :math:`i=i_hw+i_w` if ``upto_w`` is ``None``, or :math:`i=i_hw_\text{upto}+i_w` otherwise.
        Of shape ``(min(h, upto_h) * min(w, upto_w), h * w)``.
    """
    if upto_h is None:
        upto_h = h
    if upto_w is None:
        upto_w = w
    wmat_h = gen_dft_bases_1d(h)
    wmat_w = gen_dft_bases_1d(w)
    # TODO: speed it up for performance after ensuring correctness
    wmat = np.zeros((upto_h * upto_w, h * w), dtype=complex)
    for i in range(upto_h):
        for j in range(upto_w):
            w_h = wmat_h[i, :].reshape((-1, 1)) # H-by-1
            w_w = wmat_w[j, :].reshape((1, -1)) # 1-by-W
            wmat[i * upto_w + j, :] = w_h.dot(w_w).ravel()
    return wmat


def gen_real_spherical_harmonics(l, n_lat, coord_convention='colatitude-azimuth', _check_orthonormality=False):
    r"""Generates real spherical harmonics (SHs).

    See :func:`main` for example usages, including how to do both analysis and synthesis the SHs.

    Not accurate when ``n_lat`` is too small. E.g., orthonormality no longer holds when discretization is too coarse
    (small ``n_lat``), as numerical integration fails to approximate the continuous integration.

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

        _check_orthonormality (bool, optional, internal): Whether to check orthonormal or not.

    Raises:
        NotImplementedError: If the coordinate convention specified is not implemented.

    Returns:
        tuple:
            - **ymat** (*numpy.ndarray*) -- Matrix whose rows are spherical harmonics as generated by
              :func:`scipy.special.sph_harm`. When dotted with flattened image (column) vector weighted
              by ``areas_on_unit_sphere``, the :math:`i`-th row gives the coefficient for the :math:`i`-th
              harmonics, where :math:`i=l(l+1)+m`. The input signal (in the form of 2D image indexed by two angles)
              should be flattened with :meth:`numpy.ndarray.ravel`, in row-major order: the row index varies
              the slowest, and the column index the quickest. Of shape ``((l + 1) ** 2, 2 * n_lat ** 2)``.
            - **areas_on_unit_sphere** (*numpy.ndarray*) -- Area of the unit sphere covered by each sample point.
              This is proportional to sine of colatitude and has nothing to do with azimuth/longitude.
              Used as weights for discrete summation to approximate continuous integration. Necessary in SH analysis.
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
    if coord_convention == 'colatitude-azimuth':
        azis, colats = np.meshgrid(
            np.linspace(0, 2 * np.pi, num=2 * n_lat, endpoint=False),
            np.linspace(0, np.pi, num=n_lat, endpoint=True))
    elif coord_convention == 'latitude-longitude':
        lngs, lats = np.meshgrid(
            np.linspace(-np.pi, np.pi, num=2 * n_lat, endpoint=False),
            np.linspace(np.pi / 2, -np.pi / 2, num=n_lat, endpoint=True))
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

    # Verify orthonormality of SHs
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
    if func_name == 'pca':
        pts = np.random.rand(5, 8) # 8 points in 5D
        # Find all principal components
        n_pcs = pts.shape[0] - 1
        _, pcs, projs, data_mean = pca(pts, n_pcs=n_pcs)
        # Reconstruct data with only the top two PC's
        k = 2
        pts_recon = pcs[:, :k].dot(projs[:k, :]) + \
            np.tile(data_mean, (projs.shape[1], 1)).T
        print("Recon:")
        print(pts_recon)

    elif func_name == 'gen_dft_bases_1d':
        signal = np.random.randint(0, 255, 10)
        n = len(signal)
        # Transform by my matrix
        dft_mat = gen_dft_bases_1d(n)
        coeffs = dft_mat.dot(signal)
        # Transform by numpy
        coeffs_np = np.fft.fft(signal) / np.sqrt(n)
        print("Max. magnitude difference: %e" % np.abs(coeffs - coeffs_np).max())

    elif func_name == 'gen_dft_bases_2d':
        from os import environ
        from os.path import join
        import cv2
        from visualization import matrix_as_heatmap_complex
        im = np.random.randint(0, 255, (64, 128))
        h, w = im.shape
        # Transform by my matrix
        im_1d = im.ravel()
        dft_mat = gen_dft_bases_2d(h, w)
        coeffs = dft_mat.dot(im_1d)
        coeffs = coeffs.reshape((h, w))
        # Transform by numpy
        coeffs_np = np.fft.fft2(im) / (np.sqrt(h) * np.sqrt(w))
        tmp_dir = environ['TMP_DIR']
        matrix_as_heatmap_complex(coeffs, outpath=join(tmp_dir, 'coeffs_mine.png'))
        matrix_as_heatmap_complex(coeffs_np, outpath=join(tmp_dir, 'coeffs_np.png'))
        print("Max. magnitude difference: %e" % np.abs(coeffs - coeffs_np).max())
        # Reconstruct
        recon_mine = dft_mat.dot(coeffs.ravel()).reshape((h, w))
        recon_np = dft_mat.dot(coeffs_np.ravel()).reshape((h, w))
        assert np.allclose(np.imag(recon_mine), 0)
        assert np.allclose(np.imag(recon_np), 0)
        recon_mine = np.real(recon_mine)
        recon_np = np.real(recon_np)
        cv2.imwrite(join(tmp_dir, 'orig.png'), im)
        cv2.imwrite(join(tmp_dir, 'recon_mine.png'), recon_mine)
        cv2.imwrite(join(tmp_dir, 'recon_np.png'), recon_np)

    elif func_name == 'gen_real_spherical_harmonics':
        from os import environ
        from os.path import join
        from visualization import matrix_as_heatmap
        ls = [1, 2, 3, 4]
        n_steps_theta = 64
        for l in ls:
            print("l = %d" % l)
            # Generata harmonics
            ymat, weights = gen_real_spherical_harmonics(
                l, n_steps_theta, _check_orthonormality=False
            )
            # Black background with white signal
            coeffs_gt = np.random.random(ymat.shape[0])
            sph_func_1d = None
            for ci, c in enumerate(coeffs_gt):
                y_lm = ymat[ci, :]
                if sph_func_1d is None:
                    sph_func_1d = c * y_lm
                else:
                    sph_func_1d += c * y_lm
            sph_func = sph_func_1d.reshape((n_steps_theta, 2 * n_steps_theta))
            sph_func_ravel = sph_func.ravel()
            assert (sph_func_1d == sph_func_ravel).all()
            tmp_dir = environ['TMP_DIR']
            matrix_as_heatmap(sph_func, outpath=join(tmp_dir, 'sph_orig.png'))
            # Analysis
            coeffs = ymat.dot(np.multiply(weights, sph_func_ravel))
            print("\tGT")
            print(coeffs_gt)
            print("\tRecon")
            print(coeffs)
            # Synthesis
            sph_func_1d_recon = ymat.T.dot(coeffs)
            sph_func_recon = sph_func_1d_recon.reshape(sph_func.shape)
            print("Max. magnitude difference: %e" % np.abs(sph_func_1d - sph_func_1d_recon).max())
            matrix_as_heatmap(sph_func_recon, outpath=join(tmp_dir, 'sph_recon_l%03d.png' % l))

    else:
        raise NotImplementedError("Unit tests for %s" % func_name)

    embed()


if __name__ == '__main__':
    from argparse import ArgumentParser
    from IPython import embed

    parser = ArgumentParser()
    parser.add_argument('func', type=str, help="function to test")
    args = parser.parse_args()

    main(args.func)

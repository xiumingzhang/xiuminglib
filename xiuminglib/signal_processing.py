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


def dft_1d_bases(n, upto=None):
    """Generates 1D discrete Fourier transform (DFT) bases.

    Bases are rows of :math:`Y`, unitary: :math:`Y^*Y=YY^*=I`, where :math:`Y^*` is the
    conjugate transpose, and symmetric. The forward process (analysis) is :math:`X=Yx`,
    and the inverse (synthesis) is :math:`x=Y^{-1}X=Y^*X`.

    See :func:`main` for example usages.

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


def dft_2d_freq(h, w):
    """Gets 2D discrete Fourier transform (DFT) sample frequencies.

    See :func:`dft_2d_bases_real` for how this is useful for generating basis images.

    Args:
        h (int): Image height.
        w

    Returns:
        tuple:
            - **freq_h** (*numpy.ndarray*) -- Sample frequencies, in cycles per pixel, along the height
              dimension. E.g., if ``freq_h[i, j] == 0.5``, then the ``(i, j)``-th component repeats
              every 2 pixels along the height dimension.
            - **freq_w**
    """
    freq_h = np.fft.fftfreq(h)
    freq_w = np.fft.fftfreq(w)
    freq_h, freq_w = np.meshgrid(freq_h, freq_w, indexing='ij')
    return freq_h, freq_w


def dft_2d_bases(h, w, upto_h=None, upto_w=None):
    r"""Generates bases for 2D discrete Fourier transform (DFT).

    Bases are rows of :math:`Y_h` and :math:`Y_w`. See :func:`dft_1d_bases` for matrix properties.

    Input image :math:`x` should be transformed by both matrices (i.e., along both dimensions).
    Specifically, the analysis process is :math:`X=Y_hxY_w`, and the synthesis process is :math:`x=Y_h^*XY_w^*`.
    See :func:`main` for example usages.

    Args:
        h (int): Image height.
        w
        upto_h (int, optional): Up to how many bases in the height dimension. ``None`` means all.
        upto_w

    Returns:
        tuple:
            - **dft_mat_h** (*numpy.ndarray*) -- DFT matrix :math:`Y_h` transforming rows
              of the 2D signal. Of shape ``(min(h, upto_h), h)``.
            - **dft_mat_w** (*numpy.ndarray*) -- :math:`Y_w` transforming columns. Of shape
              ``(w, min(w, upto_w))``.
    """
    dft_mat_h = dft_1d_bases(h, upto=upto_h)
    dft_mat_w = dft_1d_bases(w, upto=upto_w) # shape: (upto_w, w)
    dft_mat_w = dft_mat_w.T # because it's symmetric
    return dft_mat_h, dft_mat_w


def dft_2d_bases_real(h, w, upto_h=None, upto_w=None):
    """Generates discrete Fourier transform (DFT) basis images, with which real DFT can be done.

    Unlike :func:`dft_2d_bases`, this function generates a single matrix, whose rows are flattened
    DFT basis images (defined as strictly real 2D spatial signals that, when DFT'ed, lead to 1s at
    just one sample frequency [but usually mapped to multiple entries in the coefficient matrix]).

    Using the DFT property that a real signal leads to "mirrored" DFT coefficients,
    this function first constructs such mirrored coefficients and then does synthesis with them
    to produce a basis image.

    See Also:
        ``A[1:n/2]`` contains the positive-frequency terms, and ``A[n/2+1:]`` contains the
        negative-frequency terms, in order of decreasingly negative frequency. For an even number
        of input points, ``A[n/2]`` represents both positive and negative Nyquist frequency, and
        is also purely real for real input. For an odd number of input points, ``A[(n-1)/2]``
        contains the largest positive frequency, while ``A[(n+1)/2]`` contains the largest negative
        frequency. Source: :mod:`numpy.fft`.

    The matrix of the generated bases :math:`Y` can also be used to perform "real DFT,"
    as it has been made orthonormal by careful normalization. Denote the :func:`numpy.ndarray.ravel`'ed
    image by :math:`x`. Analysis: :math:`X=Yx`. Synthesis: :math:`x=Y^{-1}X=Y^TX`. The results are the same
    as if we used the two matrices returned by :func:`dft_2d_bases`. See :func:`main` for examples.

    Args:
        h (int): Image height.
        w
        upto_h (int, optional): Up to how many bases in the height dimension. ``None`` means all.
        upto_w

    Returns:
        numpy.ndarray: Matrix with flattened basis images as rows. Row ``k``, when
        :func:`numpy.ndarray.reshape`'ed into ``(h, w)``, is the ``(i, j)``-th frequency
        component, where ``k = i * min(w, upto_w) + j``. Of shape
        ``(min(h, upto_h) * min(w, upto_w), h * w)``.
    """
    if upto_h is None:
        upto_h = h
    if upto_w is None:
        upto_w = w
    freq_h, freq_w = dft_2d_freq(h, w)
    freq_h_abs = abs(freq_h)
    freq_w_abs = abs(freq_w)
    dft_mat_h, dft_mat_w = dft_2d_bases(h, w)
    dft_real_mat = np.zeros((upto_h * upto_w, h * w))
    for i in range(upto_h):
        for j in range(upto_w):
            # Set correct entries to 1s
            f_h = freq_h_abs[i, j]
            f_w = freq_w_abs[i, j]
            is_relevant = (np.abs(freq_h_abs) == f_h) & (np.abs(freq_w_abs) == f_w)
            coeffs = np.zeros((h, w))
            if is_relevant.any():
                coeffs[is_relevant] = 1 / np.sum(is_relevant) # to make orthonormal
            else: # DC
                pass # doesn't matter anyways
            # Synthesis
            img = dft_mat_h.conj().dot(coeffs).dot(dft_mat_w.conj())
            # Essentially real
            assert np.allclose(np.imag(img), 0)
            img = np.real(img)
            # Flatten and put it into matrix
            k = i * upto_w + j
            dft_real_mat[k, :] = img.ravel()
    return dft_real_mat


def sh_bases_real(l, n_lat, coord_convention='colatitude-azimuth', _check_orthonormality=False):
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

    elif func_name == 'dft_1d_bases':
        signal = np.random.randint(0, 255, 10)
        n = len(signal)
        # Transform by my matrix
        dft_mat = dft_1d_bases(n)
        coeffs = dft_mat.dot(signal)
        # Transform by numpy
        coeffs_np = np.fft.fft(signal, norm='ortho')
        print("Max. magnitude difference: %e" % np.abs(coeffs - coeffs_np).max())

    elif func_name == 'dft_2d_freq':
        h, w = 4, 8
        freq_h, freq_w = dft_2d_freq(h, w)
        print("Along height:")
        print(freq_h)
        print("Along width:")
        print(freq_w)

    elif func_name.startswith('dft_2d_bases'):
        from os.path import join
        import cv2
        import xiuminglib as xlib
        im = np.zeros((64, 128))
        for j in np.linspace(0, im.shape[1], 8, endpoint=False):
            im[:, int(j)] = 255
        for i in np.linspace(0, im.shape[0], 8, endpoint=False):
            im[int(i), :] = 255
        h, w = im.shape
        tmp_dir = xlib.constants['dir_tmp']
        dft_mat_h, dft_mat_w = dft_2d_bases(h, w)
        if not func_name.endswith('_real'):
            from visualization import matrix_as_heatmap_complex
            # Transform by my matrix
            coeffs = dft_mat_h.dot(im).dot(dft_mat_w)
            # Transform by numpy
            coeffs_np = np.fft.fft2(im, norm='ortho')
            matrix_as_heatmap_complex(coeffs, outpath=join(tmp_dir, 'coeffs_mine.png'))
            matrix_as_heatmap_complex(coeffs_np, outpath=join(tmp_dir, 'coeffs_np.png'))
            print("(Ours 2D vs. NumPy)\tMax. mag. diff.:\t%e" % np.abs(coeffs - coeffs_np).max())
            # Reconstruct
            recon = dft_mat_h.conj().dot(coeffs).dot(dft_mat_w.conj())
            assert np.allclose(np.imag(recon), 0)
            recon = np.real(recon)
            cv2.imwrite(join(tmp_dir, 'orig.png'), im)
            cv2.imwrite(join(tmp_dir, 'recon.png'), recon)
        else:
            from general import makedirs
            from visualization import matrix_as_image
            dft_real_mat = dft_2d_bases_real(h, w)
            # Visualize bases
            out_dir = join(tmp_dir, 'real-dft')
            makedirs(out_dir, rm_if_exists=True)
            for k in range(dft_real_mat.shape[0]):
                i = k // w
                j = k - i * w
                out_f = join(out_dir, 'i%03d_j%03d.png' % (i, j))
                img = dft_real_mat[k, :].reshape((h, w))
                matrix_as_image(img, outpath=out_f)
            # Real DFT
            im_1d = im.ravel()
            coeffs = dft_real_mat.dot(im_1d)
            recon_1d = dft_real_mat.T.dot(coeffs)
            recon = recon_1d.reshape(im.shape)
            print("(Ours Real vs. GT)\t\tRecon.\tMax. mag. diff.:\t%e" % np.abs(im - recon).max())
            cv2.imwrite(join(tmp_dir, 'orig.png'), im)
            cv2.imwrite(join(tmp_dir, 'recon.png'), recon)
            coeffs_ = dft_mat_h.dot(im).dot(dft_mat_w)
            coeffs_ = coeffs_.ravel()
            print("(Ours Real vs. Ours Twice)\tCoeff.\tMax. mag. diff.:\t%e" % np.abs(coeffs - coeffs_).max())

    elif func_name == 'cameraman_dft':
        from os.path import join
        import cv2
        import xiuminglib as xlib
        im = cv2.imread(xlib.constants['path_cameraman'], cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (64, 64))
        # My DFT
        dft_real_mat = dft_2d_bases_real(*im.shape)
        coeffs = dft_real_mat.dot(im.ravel())
        recon = dft_real_mat.T.dot(coeffs).reshape(im.shape)
        cv2.imwrite(join(xlib.constants['dir_tmp'], 'a.png'), recon.astype(im.dtype))

    elif func_name == 'sh_bases_real':
        from os.path import join
        from visualization import matrix_as_heatmap
        import xiuminglib as xlib
        ls = [1, 2, 3, 4]
        n_steps_theta = 64
        for l in ls:
            print("l = %d" % l)
            # Generata harmonics
            ymat, weights = sh_bases_real(
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
            tmp_dir = xlib.constants['dir_tmp']
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


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('func', type=str, help="function to test")
    args = parser.parse_args()

    main(args.func)

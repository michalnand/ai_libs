import numpy 

import scipy.signal.windows

def tukey_window(x: numpy.ndarray) -> numpy.ndarray:
    """
    Apply tukey window independently to each feature column.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: (window_size, num_features)

    Returns
    -------
    numpy.ndarray
        Windowed signal with same shape as input.
    """ 

    window_size = x.shape[0]

    # tukey window
    #window = numpy.hanning(window_size)
    window = scipy.signal.windows.tukey(window_size, alpha=0.5, sym=True)

    # Expand to broadcast across features
    window = window[:, numpy.newaxis]

    return x * window


def fft(x: numpy.ndarray):
    """
    Compute FFT for each feature column.

    Parameters
    ----------
    x : numpy.ndarray
        Shape: (window_size, num_features)

    Returns
    -------
    fft_real : np.ndarray
        Real FFT coefficients.
        Shape: (freq_bins, num_features)

    fft_imag : np.ndarray
        Imaginary FFT coefficients.
        Shape: (freq_bins, num_features)
    """

    # FFT along time axis
    fft_complex = numpy.fft.rfft(x, axis=0)

    fft_real = fft_complex.real
    fft_imag = fft_complex.imag

    return fft_real, fft_imag

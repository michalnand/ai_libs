import numpy


class BiquadFilter:
    """
    Generic discrete biquad (2nd order IIR) filter.

    Transfer function:
        H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)

    The filter is implemented using the Direct Form II Transposed structure.
    """

    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float):
        """
        Parameters
        ----------
        b0, b1, b2 : float
            Numerator (feedforward) coefficients.
        a1, a2 : float
            Denominator (feedback) coefficients.
            Note: a0 is assumed to be 1.
        """
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.reset()

    def reset(self):
        """Reset the filter state (delay elements)."""
        self._w1 = 0.0
        self._w2 = 0.0

    def process(self, x: float) -> float:
        """
        Process a single input sample through the filter.

        Uses Direct Form II Transposed:
            y[n] = b0*x[n] + w1[n-1]
            w1[n] = b1*x[n] - a1*y[n] + w2[n-1]
            w2[n] = b2*x[n] - a2*y[n]
        """
        y = self.b0 * x + self._w1
        self._w1 = self.b1 * x - self.a1 * y + self._w2
        self._w2 = self.b2 * x - self.a2 * y
        return y
    

    def __call__(self, x: float) -> float:
        return self.process(x)

    def process_seq(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Process an array of input samples.

        Parameters
        ----------
        x : numpy.ndarray
            Input signal array.

        Returns
        -------
        numpy.ndarray
            Filtered output signal.
        """
        x = numpy.asarray(x, dtype=numpy.float64)
        y = numpy.empty_like(x)
        for n in range(len(x)):
            y[n] = self.process(x[n])
        return y

    def frequency_response(self, num_points: int = 512) -> tuple:
        """
        Compute the frequency response of the filter.

        Parameters
        ----------
        num_points : int
            Number of frequency points (from 0 to pi).

        Returns
        -------
        w : numpy.ndarray
            Angular frequencies (0 to pi).
        H : numpy.ndarray
            Complex frequency response.
        """
        w = numpy.linspace(0, numpy.pi, num_points)
        z = numpy.exp(1j * w)
        numerator = self.b0 + self.b1 * z**(-1) + self.b2 * z**(-2)
        denominator = 1.0 + self.a1 * z**(-1) + self.a2 * z**(-2)
        H = numerator / denominator
        return w, H

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"b0={self.b0}, b1={self.b1}, b2={self.b2}, "
            f"a1={self.a1}, a2={self.a2})"
        )


class LowPassFilter(BiquadFilter):
    """
    2nd order Butterworth low-pass biquad filter.

    Parameters
    ----------
    w0 : float
        Cutoff frequency in radians (0, pi).
    Q : float
        Quality factor. Default is 1/sqrt(2) ≈ 0.7071 (Butterworth).
    """

    def __init__(self, w0: float, Q: float = 1.0 / numpy.sqrt(2.0)):
        alpha = numpy.sin(w0) / (2.0 * Q)
        cos_w0 = numpy.cos(w0)

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        # Normalize by a0
        super().__init__(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )
        self.w0 = w0
        self.Q = Q


class HighPassFilter(BiquadFilter):
    """
    2nd order Butterworth high-pass biquad filter.

    Parameters
    ----------
    w0 : float
        Cutoff frequency in radians (0, pi).
    Q : float
        Quality factor. Default is 1/sqrt(2) ≈ 0.7071 (Butterworth).
    """

    def __init__(self, w0: float, Q: float = 1.0 / numpy.sqrt(2.0)):
        alpha = numpy.sin(w0) / (2.0 * Q)
        cos_w0 = numpy.cos(w0)

        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        # Normalize by a0
        super().__init__(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )
        self.w0 = w0
        self.Q = Q


class BandPassFilter(BiquadFilter):
    """
    2nd order band-pass biquad filter (constant 0 dB peak gain).

    Parameters
    ----------
    w0 : float
        Center frequency in radians (0, pi).
    Q : float
        Quality factor. Controls bandwidth (BW ≈ w0/Q).
    """

    def __init__(self, w0: float, Q: float = 1.0):
        alpha = numpy.sin(w0) / (2.0 * Q)
        cos_w0 = numpy.cos(w0)

        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        # Normalize by a0
        super().__init__(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )
        self.w0 = w0
        self.Q = Q


class BandStopFilter(BiquadFilter):
    """
    2nd order band-stop (notch) biquad filter.

    Parameters
    ----------
    w0 : float
        Center (notch) frequency in radians (0, pi).
    Q : float
        Quality factor. Controls the notch bandwidth (BW ≈ w0/Q).
    """

    def __init__(self, w0: float, Q: float = 1.0):
        alpha = numpy.sin(w0) / (2.0 * Q)
        cos_w0 = numpy.cos(w0)

        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        # Normalize by a0
        super().__init__(
            b0=b0 / a0,
            b1=b1 / a0,
            b2=b2 / a0,
            a1=a1 / a0,
            a2=a2 / a0,
        )
        self.w0 = w0
        self.Q = Q